import numpy as np

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels

# 1 nm RMS of tip/tilt mode amplitude = 4e-9/38.5 rad on sky, at 7.5 mas/px
NM_TO_PX = (4e-9 / 38.5 * 3600 * 180 / np.pi * 1e3) / 7.5


class SpotSupervisor(BaseProcessingObj):
    """
    Step-1 supervisor for spot (re)acquisition of a single-subaperture WCoG tip/tilt sensor.

    A matched filter (core + halo) runs on every frame; each peak position is registered with the
    applied tip/tilt command (disturbance coordinates), so that the spot motion caused by our own
    correction does not spoil the consensus. When the last ``n_cons`` estimates agree (within
    ``delta_factor`` * sigma_w) and lie farther than that from the WCoG window, the spot is lost:

    * ``hold`` (default): the window (and the slope zero of :class:`ShSlopecMovable`) is put on the
      estimated spot position, so the loop stabilises the spot where it is; after ``k_hold`` frames
      a feedforward tip/tilt command equal to the window centre is issued and the window returns to
      the reference (one shot);
    * ``one``: window to the reference and feedforward by the estimate at once;
    * ``n3``: as ``hold`` but the return is done in three steps, 2 frames apart.

    With ``look_frames`` = M > 1 each consensus estimate is the peak of the mean of M frames registered
    with the applied command (needed when a single frame is too faint: the matched-filter peak of one
    frame is pure noise at the faint end). ``search_radius`` > 0 restricts the peak search (and the
    guard) to a disk of that radius (px) around the reference; 0 = whole frame.

    Guard C vetoes a move if the correlation-map peak near the window is at least ``q_thr`` of the
    global peak (the window already captures the spot). Optional presence detection (registered
    blocks of ``block_frames`` frames, threshold ``z_thr`` on (peak - mean) / std of the correlation
    map) declares a dropout after ``k_absent`` absent blocks: the ``hold`` flag of ``out_window`` is
    set (:class:`ShSlopecMovable` then outputs zero slopes) and nothing is moved until ``k_present``
    present blocks.

    Combined presence (optional, ``z_thr_local`` and/or ``flux_thr`` set): a block counts as "present"
    if ANY of the global z > ``z_thr``, the local z (correlation peak within ``r_loc_sigma`` sigma_w of the
    window, same normalisation) > ``z_thr_local``, or the window flux (Gaussian WCoG mask on the window,
    per-frame clip at 0 -- ShSlopec's ``subap_tot`` for ``thr_value = 0``) > ``flux_thr``. A dropout is
    therefore declared only when all three evidences are absent (a faint star still in the window keeps
    the loop closed; a spot lost elsewhere keeps the global z up). The global z alone has a high
    noise floor (maximum over the whole field) and made faint-but-present stars look absent.
    ``presence_register`` (default True, the original behaviour) registers presence blocks on the
    command; during normal tracking the spot is stationary on the detector, so False is preferable.

    ``confirm`` (``hold``/``n3`` only): after the window move, a block of ``confirm_frames`` frames at the
    new window must show local z > ``z_thr_local`` or window flux > ``flux_thr`` before the feedforward is
    applied; otherwise the move is aborted and the window restored (a wrong consensus then costs nothing
    more than the hold). Thresholds are shared with presence, so ``confirm_frames`` defaults to
    ``block_frames``.

    ``ref_offset`` is the offset of the matched-filter peak from the WCoG centroid for a spot at the
    reference (peak/centroid asymmetry, integer-pixel quantisation): the estimates are the centroid
    displacement from the reference, so the window centre ``w`` needs no further correction.

    Coordinates: x = column, y = row, in pixels from the subaperture centre. ``in_command`` is the
    tip/tilt command really applied during the frame (nm); ``cmd_to_px`` maps it to the spot shift
    produced by the correction (px per nm, sign and axes to be calibrated on the system).

    Outputs: ``out_window`` = [wx, wy, hold] (feed it to ShSlopecMovable, delayed by one step),
    ``out_feedforward`` = accumulated feedforward command (nm, feed it to an extra tip/tilt DM),
    ``out_state`` = telemetry [est_x, est_y, z, guard_ratio, dropout, n_moves, z_local, window_flux, n_aborts]
    (z, z_local, window_flux: last completed presence block, NaN if presence is off).
    """

    def __init__(self,
                 weighted_pix_rad: float,
                 np_sub: int = 240,
                 cmd_to_px: list = None,
                 ref_offset: list = None,
                 ff_mode: str = 'hold',
                 n_cons: int = 5,
                 delta_factor: float = 1.5,
                 q_thr: float = 0.6,
                 r_loc_sigma: float = 3.0,
                 k_hold: int = None,
                 tpl_fwhm: float = 2.0,
                 halo_fwhm: float = 15.0,
                 halo_fraction: float = 0.3,
                 presence: bool = False,
                 z_thr: float = None,
                 block_frames: int = 1,
                 k_absent: int = 8,
                 k_present: int = 3,
                 look_frames: int = 1,
                 search_radius: float = 0.0,
                 z_thr_local: float = None,
                 flux_thr: float = None,
                 presence_register: bool = True,
                 confirm: bool = False,
                 confirm_frames: int = None,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if ff_mode not in ('hold', 'one', 'n3'):
            raise ValueError(f"ff_mode must be 'hold', 'one' or 'n3', got {ff_mode!r}")
        if presence and z_thr is None:
            raise ValueError('presence detection needs z_thr (noise-only quantile of the correlation z-score)')
        self.n = int(np_sub)
        self.sigma_w = 2.0 * weighted_pix_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        self.ff_mode = ff_mode
        self.n_cons = int(n_cons)
        self.delta = delta_factor * self.sigma_w
        self.q_thr = q_thr
        self.r_loc2 = (r_loc_sigma * self.sigma_w) ** 2
        self.k_hold = self.n_cons if k_hold is None else int(k_hold)
        self.presence = presence
        self.z_thr = z_thr
        self.block_frames = int(block_frames)
        self.k_absent = int(k_absent)
        self.k_present = int(k_present)
        self.look_frames = int(look_frames)
        self.search_radius = float(search_radius)
        self.z_thr_local = z_thr_local
        self.flux_thr = flux_thr
        self.presence_register = bool(presence_register)
        self.confirm = bool(confirm)
        self.confirm_frames = self.block_frames if confirm_frames is None else int(confirm_frames)
        if self.confirm and z_thr_local is None and flux_thr is None:
            raise ValueError('confirm needs z_thr_local and/or flux_thr (local evidence at the new window)')
        if self.confirm and ff_mode == 'one':
            raise ValueError("confirm needs a hold phase: use ff_mode 'hold' or 'n3'")

        cmd = np.asarray(cmd_to_px if cmd_to_px is not None else np.eye(2) * NM_TO_PX, dtype=float)
        self.cmd_to_px = cmd.reshape(2, 2)
        self.px_to_cmd = np.linalg.inv(self.cmd_to_px)
        self.ref_offset = np.asarray(ref_offset if ref_offset is not None else [0.0, 0.0], dtype=float)

        # matched filter: core + halo template, peak at index 0 (wrapped), so that the
        # correlation peak index of a spot equals its pixel index
        idx = np.arange(self.n)
        wrapped = np.minimum(idx, self.n - idx)
        r2 = wrapped[:, None] ** 2 + wrapped[None, :] ** 2

        def gauss(fwhm):
            g = np.exp(-r2 / (2.0 * (fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))) ** 2))
            return g / g.sum()

        tpl = (1.0 - halo_fraction) * gauss(tpl_fwhm) + halo_fraction * gauss(halo_fwhm)
        self.tpl_conj = self.xp.asarray(np.conj(np.fft.fft2(tpl)).astype(self.complex_dtype))
        self.c = (self.n - 1) / 2.0
        self.coord = idx - self.c                                  # pixel index -> px from the centre
        fr = np.fft.fftfreq(self.n)
        self.fx, self.fy = fr[None, :], fr[:, None]

        self.n_moves = 0
        self.n_aborts = 0
        self._mask_w = None
        self.reset_state()

        self.inputs['in_pixels'] = InputValue(type=Pixels)
        self.inputs['in_command'] = InputValue(type=BaseValue)
        self.window = BaseValue(value=self.xp.zeros(3, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self.feedforward = BaseValue(value=self.xp.zeros(2, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self.state_out = BaseValue(value=self.xp.zeros(9, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self.outputs['out_window'] = self.window
        self.outputs['out_feedforward'] = self.feedforward
        self.outputs['out_state'] = self.state_out

    def reset_state(self):
        self.w = np.zeros(2)                       # window centre, px
        self.u_ff = np.zeros(2)                    # accumulated feedforward, px (spot shift)
        self.hist, self.ratios = [], []
        self.hold_until = None
        self.ff_pending = None
        self.n3_left, self.n3_next, self.n3_w0 = 0, 0, np.zeros(2)
        self.dropout = False
        self.hit = self.miss = 0
        self.acc = None
        self.acc_count = 0
        self.u_block0 = np.zeros(2)
        self.look_acc = None
        self.look_count = 0
        self.u_look0 = np.zeros(2)
        self.acc_flux = 0.0
        self.last_block = (np.nan, np.nan, np.nan)      # global z, local z, window flux of the last block
        self.w_prev = np.zeros(2)
        self.conf_active = False
        self.conf_count = 0
        self.conf_acc = None
        self.conf_flux = 0.0
        self.confirmed = None
        self.frame = 0

    @classmethod
    def input_names(cls):
        return {'in_pixels': InputDesc(Pixels, 'Detector pixels'),
                'in_command': InputDesc(BaseValue, 'Tip/tilt command applied during the frame [nm]')}

    @classmethod
    def output_names(cls):
        return {'out_window': OutputDesc(BaseValue, 'Window centre [wx, wy] px and hold flag'),
                'out_feedforward': OutputDesc(BaseValue, 'Accumulated feedforward tip/tilt command [nm]'),
                'out_state': OutputDesc(BaseValue, 'Telemetry [est_x, est_y, z, guard_ratio, dropout, n_moves, '
                                                   'z_local, window_flux, n_aborts]')}

    # ---- algorithm (host-side decisions on device correlation maps) ----

    def _correlate(self, spectrum):
        return self.xp.real(self.xp.fft.ifft2(spectrum * self.tpl_conj))

    def _restrict(self, corr):
        if self.search_radius <= 0:
            return corr
        if not hasattr(self, '_search_mask'):
            g = (self.coord - self.ref_offset[0]) ** 2, (self.coord - self.ref_offset[1]) ** 2
            self._search_mask = self.xp.asarray((g[1][:, None] + g[0][None, :]) <= self.search_radius ** 2)
        return self.xp.where(self._search_mask, corr, corr.min())

    def _peak(self, corr):
        i = int(self.xp.argmax(corr))
        row, col = divmod(i, self.n)
        return np.array([col - self.c, row - self.c]) - self.ref_offset

    def _guard_ratio(self, corr):
        gx = (self.coord - self.ref_offset[0] - self.w[0]) ** 2
        gy = (self.coord - self.ref_offset[1] - self.w[1]) ** 2
        mask = self.xp.asarray((gy[:, None] + gx[None, :]) <= self.r_loc2)
        if not bool(mask.any()):
            return 1.0
        gl = float(corr.max())
        return float(corr[mask].max()) / gl if gl > 0 else 1.0

    def _window_mask(self):
        """Gaussian WCoG weight mask centred on the current window (as ShSlopecMovable), cached per w."""
        key = (float(self.w[0]), float(self.w[1]))
        if self._mask_w is None or self._mask_w[0] != key:
            cols = np.exp(-0.5 * ((self.coord - key[0]) / self.sigma_w) ** 2)
            rows = np.exp(-0.5 * ((self.coord - key[1]) / self.sigma_w) ** 2)
            m = np.outer(rows, cols)
            m /= m.max()
            m[m < 1e-6] = 0.0
            self._mask_w = (key, self.xp.asarray(m.astype(self.dtype)))
        return self._mask_w[1]

    def _window_flux(self, frame):
        return float(self.xp.sum(self.xp.clip(frame, 0, None) * self._window_mask()))

    def _local_disk(self):
        gx = (self.coord - self.ref_offset[0] - self.w[0]) ** 2
        gy = (self.coord - self.ref_offset[1] - self.w[1]) ** 2
        return self.xp.asarray((gy[:, None] + gx[None, :]) <= self.r_loc2)

    def _block_stats(self, corr):
        m, sd = corr.mean(), corr.std()
        disk = self._local_disk()
        z_loc = float((corr[disk].max() - m) / sd) if bool(disk.any()) else -np.inf
        return float((corr.max() - m) / sd), z_loc

    def _is_present(self, z, z_loc, flux):
        return (z > self.z_thr or (self.z_thr_local is not None and z_loc > self.z_thr_local)
                or (self.flux_thr is not None and flux > self.flux_thr))

    def _update_presence(self, spectrum, corr, u, flux):
        if self.acc_count == 0:
            self.u_block0 = u.copy()
            self.acc = self.xp.zeros_like(spectrum)
            self.acc_flux = 0.0
        if self.presence_register:
            du = u - self.u_block0
            phase = np.exp(-2j * np.pi * (self.fx * du[0] + self.fy * du[1]))
            self.acc += spectrum * self.xp.asarray(phase.astype(self.complex_dtype))
        else:
            self.acc += spectrum
        self.acc_flux += flux
        self.acc_count += 1
        if self.acc_count < self.block_frames:
            return None
        cp = corr if self.block_frames == 1 else self._correlate(self.acc / self.block_frames)
        z, z_loc = self._block_stats(cp)
        f = self.acc_flux / self.block_frames
        self.last_block = (z, z_loc, f)
        self.acc_count = 0
        if self._is_present(z, z_loc, f):
            self.hit, self.miss = self.hit + 1, 0
        else:
            self.hit, self.miss = 0, self.miss + 1
        if not self.dropout and self.miss >= self.k_absent:
            self.dropout = True
            self.look_count = 0
            if self.conf_active:                  # restart the confirmation from fresh frames after release
                self.conf_count, self.conf_acc, self.conf_flux, self.confirmed = 0, None, 0.0, None
        elif self.dropout and self.hit >= self.k_present:
            self.dropout = False
            self.hist, self.ratios = [], []
            self.look_count = 0
        return z

    def process_frame(self, frame, u_loop_px):
        """One detector frame. ``u_loop_px``: spot shift (px) produced by the loop command applied
        during the frame; the supervisor adds its own feedforward. Returns a telemetry dict."""
        u = np.asarray(u_loop_px, dtype=float) + self.u_ff
        frame = self.xp.asarray(frame, dtype=self.dtype)
        spectrum = self.xp.fft.fft2(frame)
        corr = self._correlate(spectrum)
        need_flux = (self.presence and self.flux_thr is not None) or (self.conf_active and self.flux_thr is not None)
        flux = self._window_flux(frame) if need_flux else 0.0
        if self.conf_active and not self.dropout:            # no evidence is collected while blind
            self._update_confirmation(spectrum, flux)
        z = self._update_presence(spectrum, corr, u, flux) if self.presence else None
        est = np.full(2, np.nan)
        ratio = np.nan
        look = self._look(spectrum, corr, u) if not self.dropout else None
        if look is not None:
            corr_l, u0 = look
            corr_l = self._restrict(corr_l)
            est = self._peak(corr_l)
            self.hist.append(est + u0)
            ratio = self._guard_ratio(corr_l)
            self.ratios.append(ratio)
            self.hist, self.ratios = self.hist[-self.n_cons:], self.ratios[-self.n_cons:]
            if len(self.hist) == self.n_cons:
                arr = np.array(self.hist)
                med = np.median(arr, axis=0)
                far = np.linalg.norm(med - (self.w + u)) > self.delta
                agree = np.linalg.norm(arr - med, axis=1).max() <= self.delta
                if agree and far and np.median(self.ratios) < self.q_thr:
                    self._start_move(med - u)
        self._schedule_feedforward()
        self.frame += 1
        return dict(est=est, z=z, ratio=ratio, dropout=self.dropout)

    def _look(self, spectrum, corr, u):
        """Correlation map of one look: the frame itself (look_frames = 1) or the mean of look_frames
        frames registered to the command of the first one; None while the look is incomplete."""
        if self.look_frames == 1:
            return corr, u
        if self.look_count == 0:
            self.u_look0 = u.copy()
            self.look_acc = self.xp.zeros_like(spectrum)
        du = u - self.u_look0
        phase = np.exp(-2j * np.pi * (self.fx * du[0] + self.fy * du[1]))
        self.look_acc += spectrum * self.xp.asarray(phase.astype(self.complex_dtype))
        self.look_count += 1
        if self.look_count < self.look_frames:
            return None
        self.look_count = 0
        return self._correlate(self.look_acc / self.look_frames), self.u_look0

    def _update_confirmation(self, spectrum, flux):
        """Accumulate frames at the new window during the hold; decide once confirm_frames are in."""
        if self.confirmed is not None:
            return
        self.conf_acc = spectrum if self.conf_acc is None else self.conf_acc + spectrum
        self.conf_flux += flux
        self.conf_count += 1
        if self.conf_count < self.confirm_frames:
            return
        corr = self._correlate(self.conf_acc / self.conf_count)
        _, z_loc = self._block_stats(corr)
        f = self.conf_flux / self.conf_count
        self.confirmed = ((self.z_thr_local is not None and z_loc > self.z_thr_local)
                          or (self.flux_thr is not None and f > self.flux_thr))

    def _start_move(self, xhat):
        """Consensus accepted; ``xhat``: spot position in the detector frame (px)."""
        self.n_moves += 1
        self.hist, self.ratios = [], []
        self.n3_left = 0
        if self.ff_mode == 'one':
            self.ff_pending, self.w = xhat.copy(), np.zeros(2)
        else:
            if not self.conf_active:                  # a new move during a hold keeps the original fallback
                self.w_prev = self.w.copy()
            self.w, self.hold_until = xhat.copy(), self.frame + self.k_hold
            if self.confirm:                           # evidence collected from the next frame on
                self.conf_active, self.conf_count, self.conf_acc = True, 0, None
                self.conf_flux, self.confirmed = 0.0, None

    def _schedule_feedforward(self):
        if self.ff_pending is not None:
            self.u_ff = self.u_ff + self.ff_pending
            self.ff_pending = None
        elif self.hold_until is not None and self.frame >= self.hold_until and not self.dropout:
            if self.conf_active and self.n3_left == 0:
                if self.confirmed is None:            # confirmation block not complete yet: keep holding
                    return
                self.conf_active = False
                if not self.confirmed:                # wrong consensus: undo the window move, no feedforward
                    self.n_aborts += 1
                    self.w, self.hold_until = self.w_prev.copy(), None
                    return
            if self.ff_mode == 'hold':
                self.u_ff = self.u_ff + self.w
                self.w, self.hold_until = np.zeros(2), None
            else:
                if self.n3_left == 0:
                    self.n3_left, self.n3_next, self.n3_w0 = 3, self.frame, self.w.copy()
                if self.frame >= self.n3_next:
                    self.u_ff = self.u_ff + self.n3_w0 / 3
                    self.w = self.w - self.n3_w0 / 3
                    self.n3_left, self.n3_next = self.n3_left - 1, self.frame + 2
                    if self.n3_left == 0:
                        self.w, self.hold_until = np.zeros(2), None

    def trigger_code(self):
        frame = self.local_inputs['in_pixels'].pixels
        cmd = np.asarray(cpuArray(self.local_inputs['in_command'].value), dtype=float)
        info = self.process_frame(frame, self.cmd_to_px @ cmd[:2])
        ff_nm = self.px_to_cmd @ self.u_ff
        self.window.value[:] = self.xp.asarray([self.w[0], self.w[1], float(self.dropout)], dtype=self.dtype)
        self.feedforward.value[:] = self.xp.asarray(ff_nm, dtype=self.dtype)
        z, z_loc, f = self.last_block if self.presence else (np.nan, np.nan, np.nan)
        self.state_out.value[:] = self.xp.asarray([*info['est'], z, info['ratio'], float(self.dropout), self.n_moves,
                                                   z_loc, f, self.n_aborts], dtype=self.dtype)

    def post_trigger(self):
        super().post_trigger()
        for name in ('out_window', 'out_feedforward', 'out_state'):
            self.outputs[name].generation_time = self.current_time
