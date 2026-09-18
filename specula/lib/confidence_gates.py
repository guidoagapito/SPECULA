"""Pluggable Step-3 confidence-gate strategies for `AdaptiveShrinkageSlopec`.

Each `ConfidenceGate` computes the raw (pre-`w_smooth`-EMA) shrinkage weight
`w_raw` from the per-subaperture detector-model SNR^2 (`rho_sq`) and, for
gates that need it, a per-frame confidence `margin` (see
`AdaptiveShrinkageSlopec`'s Step 1 for how `margin` is computed). A gate
instance is built once in `AdaptiveShrinkageSlopec.__init__` and never
swapped afterwards, so dispatching to it via a plain Python attribute
(`self._gate.compute(...)`) is CUDA-graph safe: the traced computation graph
depends only on which gate class was chosen at construction time, never on
per-frame data (same reasoning already used in that class for its other
constructor-time flags, e.g. `gain_correction_enable`).

Replaces the earlier `relative_gate_enable`/`relative_gate_*` constructor
flags (2026-09-17) with a `gate_type`/`gate_params` pair, to avoid growing
the base class's constructor with a new flag family per gate design (see
RESULTS.md, "Confidence-gate redesign" section, 2026-09-17/18).
"""


class ConfidenceGate:
    """Base class. Subclasses implement `compute()`; `needs_margin` and
    `telemetry()` let `AdaptiveShrinkageSlopec` stay agnostic to which
    concrete gate is active.
    """

    #: Whether this gate consumes the per-frame `margin` signal. Checked
    #: once (constructor-time), not per frame, so branching on it in the
    #: caller is CUDA-graph safe.
    needs_margin = False

    def compute(self, xp, rho_sq, margin, eps):
        """Return `w_raw`, the pre-EMA shrinkage weight, same shape as
        `rho_sq`. `margin` is None when `needs_margin` is False.
        """
        raise NotImplementedError

    def telemetry(self):
        """Optional extra per-subaperture telemetry arrays, as
        `{name: array}`. Empty dict if the gate has none. Keys are fixed
        for a given gate class (never appear/disappear frame to frame),
        so callers can safely branch on key presence once.
        """
        return {}


class WienerGate(ConfidenceGate):
    """Classic fixed-threshold Wiener/MMSE gate: `w = rho^2 / (rho^2 + k)`.

    The default, unconditionally-safe choice -- bit-for-bit identical to
    `AdaptiveShrinkageSlopec`'s behaviour before this abstraction existed.

    Parameters
    ----------
    k_wiener : float [1]
        sigma_PSF^2 / sigma_s^2. w = 0.5 occurs at rho^2 = k_wiener.
    """

    def __init__(self, k_wiener):
        self.k_wiener = k_wiener

    def compute(self, xp, rho_sq, margin, eps):
        return rho_sq / (rho_sq + self.k_wiener)


class RelativeCeilingGate(ConfidenceGate):
    """Self-normalising gate relative to an EMA ceiling of the best
    recently-achievable `rho_sq`, instead of a fixed absolute constant
    (2026-09-17; formerly `relative_gate_enable=True`).

    `rho_sq_ceiling` tracks a per-subaperture EMA of `rho_sq`, updated ONLY
    on high-margin (high-confidence) frames -- a run of bad frames does not
    itself drag the reference down. Gate: `w = rho_sq / (rho_sq + k_rel *
    rho_sq_ceiling)`.

    Validated so far only in a toy 1D closed-loop model. Real closed-loop
    probing found it cleanly fixes one specific hard case (d30/H=19.5
    seed=3) but is uniformly harmful for d55/H=19.0 across a full
    hyperparameter grid, because `margin` (spatial/structural confidence)
    and `rho_sq` (flux/SNR quality) are not well correlated in a uniformly
    flux-starved regime -- see RESULTS.md, "Closed-loop probe results:
    works cleanly for d30/seed3, uniformly harmful for d55". Treat as
    experimental, not a general-purpose replacement for `WienerGate`.

    Parameters
    ----------
    xp : module
        Array backend (`self.xp` of the owning slopec instance).
    n_subaps : int
        Number of sub-apertures (sizes the persistent ceiling buffer).
    dtype : numpy/cupy dtype
        Dtype of the ceiling buffer (`self.dtype` of the owning slopec).
    k_wiener : float [1]
        Only used as the ceiling's default init value (see `ceiling_init`).
    k_rel : float [1]
        Dimensionless ratio in the gate's denominator, relative to
        `rho_sq_ceiling` instead of a fixed constant.
    margin_thresh : float [1]
        Confidence threshold above which a frame updates `rho_sq_ceiling`.
    ema_alpha : float [1]
        EMA smoothing factor for `rho_sq_ceiling`'s update.
    ceiling_init : float [1] or None
        Initial `rho_sq_ceiling` value before any high-confidence frame.
        None (default) uses `k_wiener`, so initial behaviour is comparable
        in scale to `WienerGate` until the ceiling adapts.
    """

    needs_margin = True

    def __init__(self, xp, n_subaps, dtype, k_wiener, k_rel=0.3,
                 margin_thresh=0.5, ema_alpha=0.05, ceiling_init=None):
        self.k_rel = k_rel
        self.margin_thresh = margin_thresh
        self.ema_alpha = ema_alpha
        init_value = k_wiener if ceiling_init is None else ceiling_init
        self.rho_sq_ceiling = xp.full(n_subaps, init_value, dtype=dtype)

    def compute(self, xp, rho_sq, margin, eps):
        do_update = margin > self.margin_thresh
        updated_ceiling = ((1.0 - self.ema_alpha) * self.rho_sq_ceiling
                            + self.ema_alpha * rho_sq)
        xp.copyto(self.rho_sq_ceiling, xp.where(do_update, updated_ceiling, self.rho_sq_ceiling))
        return rho_sq / (rho_sq + self.k_rel * self.rho_sq_ceiling + eps)

    def telemetry(self):
        return {'rho_sq_ceiling': self.rho_sq_ceiling}


class ShiftedSigmoidGate(ConfidenceGate):
    """2D confidence gate: `w = min(S_snr(rho^2), S_margin(margin))`, the
    soft-minimum of two independent logistic sigmoids (2026-09-18).

    Design history (see RESULTS.md, "Confidence-gate redesign" section):
    an initial product-of-two-steep-sigmoids proposal (both beta~4-8, i.e.
    4-8x steeper than Wiener's own slope at its inflection) was found to
    (a) chatter 3.6-6.2x more than Wiener under shot noise, and (b) risk
    compounding a collapse, since `rho_sq` and `margin` were measured to
    correlate 0.889 during a real lock-loss episode, not independent as
    the product form assumes. Both risks are addressed here by: shifting
    the SNR sigmoid's centre away from `k_wiener` (decoupling "where trust
    grows" from "the absolute noise floor") while using a much gentler
    slope on BOTH axes (the inflection-point slope `beta/(4k)` is
    independent of centre location, so a gentler beta genuinely reduces
    chattering wherever the centre sits, not just at k_wiener); and
    combining via `min()` instead of a product, so a bad-but-not-awful
    value on one axis is not double-punished by a simultaneously bad value
    on the (correlated) other axis. Toy-validated candidate:
    `boost_mult=6.0, beta_snr=0.5, margin_thresh=0.25, beta_margin=2.0`
    (net RMS improvement over Wiener across good/poor/poor+windshake toy
    regimes, chattering reduced to ~2.8-4x Wiener, no worse than Wiener
    under a severe correlated-collapse stress test) -- see RESULTS.md for
    the full numeric comparison. NOT yet validated in real closed loop.

    Parameters
    ----------
    k_wiener : float [1]
        Reference SNR scale; the SNR sigmoid is centred at
        `boost_mult * k_wiener`, not at `k_wiener` itself.
    boost_mult : float [1]
        Multiplier locating the SNR sigmoid's centre relative to
        `k_wiener`.
    beta_snr : float [1]
        SNR-sigmoid steepness, in units of "x steeper than Wiener's own
        slope at its inflection point" (1.0 matches Wiener's own slope).
    margin_thresh : float [1]
        Centre of the margin sigmoid.
    beta_margin : float [1]
        Margin-sigmoid steepness (same units as `beta_snr`).
    """

    needs_margin = True

    def __init__(self, k_wiener, boost_mult=6.0, beta_snr=0.5,
                 margin_thresh=0.25, beta_margin=2.0):
        self.rho_boost_centre = boost_mult * k_wiener
        self.alpha_snr = beta_snr / k_wiener
        self.margin_thresh = margin_thresh
        self.beta_margin = beta_margin

    def compute(self, xp, rho_sq, margin, eps):
        s_snr = 1.0 / (1.0 + xp.exp(-self.alpha_snr * (rho_sq - self.rho_boost_centre)))
        s_margin = 1.0 / (1.0 + xp.exp(-self.beta_margin * (margin - self.margin_thresh)))
        return xp.minimum(s_snr, s_margin)


_GATE_CLASSES = {
    'wiener': WienerGate,
    'relative_ceiling': RelativeCeilingGate,
    'shifted_sigmoid': ShiftedSigmoidGate,
}


def build_confidence_gate(gate_type, xp, n_subaps, dtype, k_wiener, gate_params=None):
    """Factory used by `AdaptiveShrinkageSlopec.__init__`.

    Parameters
    ----------
    gate_type : str
        One of 'wiener' (default), 'relative_ceiling', 'shifted_sigmoid'.
    xp, n_subaps, dtype, k_wiener
        Forwarded to whichever gate class needs them.
    gate_params : dict or None
        Extra keyword arguments for the chosen gate class (e.g.
        `{'boost_mult': 6.0, 'beta_snr': 0.5}` for 'shifted_sigmoid').
    """
    try:
        gate_cls = _GATE_CLASSES[gate_type]
    except KeyError:
        raise ValueError(
            f"Unknown gate_type '{gate_type}', must be one of {sorted(_GATE_CLASSES)}")

    gate_params = gate_params or {}
    if gate_cls is WienerGate:
        return WienerGate(k_wiener=k_wiener)
    elif gate_cls is RelativeCeilingGate:
        return RelativeCeilingGate(xp=xp, n_subaps=n_subaps, dtype=dtype,
                                    k_wiener=k_wiener, **gate_params)
    else:
        return ShiftedSigmoidGate(k_wiener=k_wiener, **gate_params)
