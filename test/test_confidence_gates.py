import unittest

import numpy as np

from specula.lib.confidence_gates import (
    ConfidenceGate,
    RelativeCeilingGate,
    ShiftedSigmoidGate,
    WienerGate,
    build_confidence_gate,
)


class TestWienerGate(unittest.TestCase):
    """Classic fixed-threshold Wiener gate: w = rho_sq / (rho_sq + k_wiener)."""

    def test_compute_matches_hand_computed_formula(self):
        k_wiener = 10.0
        rho_sq = np.array([0.0, 5.0, 10.0, 100.0, 1e6])
        gate = WienerGate(k_wiener=k_wiener)

        w = gate.compute(np, rho_sq, margin=None, eps=1e-12)
        expected = rho_sq / (rho_sq + k_wiener)
        np.testing.assert_allclose(w, expected, rtol=1e-12)

        # Sanity: w=0.5 exactly at rho_sq == k_wiener.
        self.assertAlmostEqual(float(w[2]), 0.5, places=12)

    def test_needs_margin_is_false(self):
        self.assertFalse(WienerGate(k_wiener=10.0).needs_margin)

    def test_telemetry_is_empty(self):
        self.assertEqual(WienerGate(k_wiener=10.0).telemetry(), {})

    def test_margin_argument_is_ignored(self):
        """WienerGate.compute must not depend on margin at all -- passing
        None (the value the caller always uses since needs_margin=False) or
        a bogus array must give identical results."""
        rho_sq = np.array([1.0, 20.0, 300.0])
        gate = WienerGate(k_wiener=10.0)
        w_none = gate.compute(np, rho_sq, margin=None, eps=1e-12)
        w_bogus = gate.compute(np, rho_sq, margin=np.array([-999.0, 999.0, 0.0]), eps=1e-12)
        np.testing.assert_array_equal(w_none, w_bogus)


class TestRelativeCeilingGate(unittest.TestCase):
    """Self-normalising gate: w = rho_sq / (rho_sq + k_rel*ceiling + eps),
    with ceiling an EMA of rho_sq updated only on high-margin frames."""

    def test_ceiling_init_defaults_to_k_wiener(self):
        gate = RelativeCeilingGate(xp=np, n_subaps=4, dtype=np.float64, k_wiener=12.5)
        np.testing.assert_array_equal(gate.rho_sq_ceiling, np.full(4, 12.5))

    def test_ceiling_init_honours_explicit_value(self):
        gate = RelativeCeilingGate(xp=np, n_subaps=4, dtype=np.float64,
                                    k_wiener=12.5, ceiling_init=3.0)
        np.testing.assert_array_equal(gate.rho_sq_ceiling, np.full(4, 3.0))

    def test_needs_margin_is_true(self):
        gate = RelativeCeilingGate(xp=np, n_subaps=2, dtype=np.float64, k_wiener=10.0)
        self.assertTrue(gate.needs_margin)

    def test_telemetry_contains_rho_sq_ceiling(self):
        gate = RelativeCeilingGate(xp=np, n_subaps=3, dtype=np.float64, k_wiener=10.0)
        telemetry = gate.telemetry()
        self.assertIn('rho_sq_ceiling', telemetry)
        np.testing.assert_array_equal(telemetry['rho_sq_ceiling'], gate.rho_sq_ceiling)

    def test_compute_updates_ceiling_and_matches_formula_on_high_margin_frame(self):
        k_wiener, k_rel, margin_thresh, ema_alpha = 10.0, 0.3, 0.5, 0.05
        gate = RelativeCeilingGate(xp=np, n_subaps=2, dtype=np.float64,
                                    k_wiener=k_wiener, k_rel=k_rel,
                                    margin_thresh=margin_thresh, ema_alpha=ema_alpha)
        rho_sq = np.array([50.0, 200.0])
        margin = np.array([0.9, 0.9])  # both above threshold -> both update
        eps = 1e-12

        ceiling_before = gate.rho_sq_ceiling.copy()
        w = gate.compute(np, rho_sq, margin, eps)

        expected_ceiling = (1.0 - ema_alpha) * ceiling_before + ema_alpha * rho_sq
        np.testing.assert_allclose(gate.rho_sq_ceiling, expected_ceiling, rtol=1e-12)

        expected_w = rho_sq / (rho_sq + k_rel * expected_ceiling + eps)
        np.testing.assert_allclose(w, expected_w, rtol=1e-9)

    def test_compute_leaves_ceiling_unchanged_on_low_margin_frame(self):
        k_wiener, margin_thresh = 10.0, 0.5
        gate = RelativeCeilingGate(xp=np, n_subaps=2, dtype=np.float64,
                                    k_wiener=k_wiener, margin_thresh=margin_thresh)
        rho_sq = np.array([50.0, 200.0])
        margin = np.array([0.1, 0.1])  # both below threshold -> no update
        ceiling_before = gate.rho_sq_ceiling.copy()

        w = gate.compute(np, rho_sq, margin, eps=1e-12)

        np.testing.assert_array_equal(gate.rho_sq_ceiling, ceiling_before)
        expected_w = rho_sq / (rho_sq + gate.k_rel * ceiling_before + 1e-12)
        np.testing.assert_allclose(w, expected_w, rtol=1e-9)

    def test_compute_updates_only_subapertures_above_threshold(self):
        """Per-subaperture independence: each element's ceiling reacts only
        to its OWN margin, verified elementwise."""
        gate = RelativeCeilingGate(xp=np, n_subaps=3, dtype=np.float64,
                                    k_wiener=10.0, margin_thresh=0.5, ema_alpha=0.1)
        rho_sq = np.array([1.0, 2.0, 3.0])
        margin = np.array([0.9, 0.1, 0.9])  # subap 1 stays put
        ceiling_before = gate.rho_sq_ceiling.copy()

        gate.compute(np, rho_sq, margin, eps=1e-12)

        self.assertNotAlmostEqual(gate.rho_sq_ceiling[0], ceiling_before[0], places=9)
        self.assertAlmostEqual(gate.rho_sq_ceiling[1], ceiling_before[1], places=12)
        self.assertNotAlmostEqual(gate.rho_sq_ceiling[2], ceiling_before[2], places=9)

    def test_ceiling_state_persists_and_mutates_in_place_across_calls(self):
        """The gate must own persistent, mutable state: repeated compute()
        calls accumulate the EMA, and the underlying array object identity
        must never change (in-place semantics, required for CUDA graph
        capture in the caller)."""
        gate = RelativeCeilingGate(xp=np, n_subaps=2, dtype=np.float64,
                                    k_wiener=10.0, margin_thresh=0.5, ema_alpha=0.2)
        ceiling_buffer_id = id(gate.rho_sq_ceiling)

        rho_sq = np.array([100.0, 100.0])
        margin = np.array([0.9, 0.9])

        values = [gate.rho_sq_ceiling.copy()]
        for _ in range(5):
            gate.compute(np, rho_sq, margin, eps=1e-12)
            self.assertEqual(id(gate.rho_sq_ceiling), ceiling_buffer_id,
                "rho_sq_ceiling was reassigned instead of mutated in place")
            values.append(gate.rho_sq_ceiling.copy())

        # Monotonic march toward rho_sq=100 from init=k_wiener=10 on every
        # high-margin call -- confirms state actually accumulates frame to
        # frame rather than resetting.
        for before, after in zip(values, values[1:]):
            self.assertTrue(np.all(after > before),
                "ceiling did not keep moving toward rho_sq across repeated calls")


class TestShiftedSigmoidGate(unittest.TestCase):
    """2D confidence gate: w = min(S_snr(rho_sq), S_margin(margin))."""

    def test_compute_matches_hand_computed_formula(self):
        k_wiener, boost_mult, beta_snr = 10.0, 6.0, 0.5
        margin_thresh, beta_margin = 0.25, 2.0
        gate = ShiftedSigmoidGate(k_wiener=k_wiener, boost_mult=boost_mult,
                                   beta_snr=beta_snr, margin_thresh=margin_thresh,
                                   beta_margin=beta_margin)
        rho_sq = np.array([0.0, 60.0, 500.0])
        margin = np.array([0.0, 0.25, 1.0])

        w = gate.compute(np, rho_sq, margin, eps=1e-12)

        alpha_snr = beta_snr / k_wiener
        rho_boost_centre = boost_mult * k_wiener
        s_snr = 1.0 / (1.0 + np.exp(-alpha_snr * (rho_sq - rho_boost_centre)))
        s_margin = 1.0 / (1.0 + np.exp(-beta_margin * (margin - margin_thresh)))
        expected = np.minimum(s_snr, s_margin)
        np.testing.assert_allclose(w, expected, rtol=1e-12)

        # At margin == margin_thresh, s_margin == 0.5 exactly.
        self.assertAlmostEqual(float(s_margin[1]), 0.5, places=12)
        # At rho_sq == boost_mult*k_wiener, s_snr == 0.5 exactly.
        self.assertAlmostEqual(float(s_snr[1]), 0.5, places=12)

    def test_needs_margin_is_true(self):
        self.assertTrue(ShiftedSigmoidGate(k_wiener=10.0).needs_margin)

    def test_telemetry_is_empty(self):
        self.assertEqual(ShiftedSigmoidGate(k_wiener=10.0).telemetry(), {})

    def test_output_is_the_soft_minimum_not_a_product(self):
        """Distinguishing property vs. a product-of-sigmoids design (see
        class docstring): with both axes at a moderate, non-saturated
        value, min() must equal the smaller of the two factors exactly,
        while a product would have been strictly smaller than either
        factor alone -- this is exactly the "bad-but-not-awful on one axis
        is not double-punished" property min() is chosen for."""
        gate = ShiftedSigmoidGate(k_wiener=10.0, boost_mult=6.0, beta_snr=0.5,
                                   margin_thresh=0.25, beta_margin=2.0)
        # rho_sq at the SNR sigmoid's own centre (s_snr == 0.5 exactly);
        # margin chosen so s_margin is clearly lower than s_snr.
        rho_sq = np.array([60.0])
        margin = np.array([-0.25])
        w = gate.compute(np, rho_sq, margin, eps=1e-12)

        s_snr = 0.5
        s_margin = 1.0 / (1.0 + np.exp(-2.0 * (margin[0] - 0.25)))
        self.assertLess(s_margin, s_snr)
        np.testing.assert_allclose(w, [s_margin], rtol=1e-9,
            err_msg="min() should have returned the smaller factor (s_margin) exactly")
        # A product would have been strictly smaller than the smaller factor alone.
        self.assertGreater(float(w[0]), s_snr * s_margin)


class TestBuildConfidenceGate(unittest.TestCase):
    """Factory dispatch for gate_type -> concrete ConfidenceGate subclass."""

    def test_wiener_dispatch(self):
        gate = build_confidence_gate('wiener', xp=np, n_subaps=4, dtype=np.float64,
                                      k_wiener=10.0)
        self.assertIsInstance(gate, WienerGate)
        self.assertEqual(gate.k_wiener, 10.0)

    def test_wiener_dispatch_ignores_gate_params(self):
        """WienerGate's constructor only takes k_wiener -- the factory must
        not forward gate_params to it (matching the class docstring's
        'Ignored (and may be omitted) for the default wiener gate')."""
        gate = build_confidence_gate('wiener', xp=np, n_subaps=4, dtype=np.float64,
                                      k_wiener=10.0,
                                      gate_params={'k_rel': 100.0, 'bogus': 1})
        self.assertIsInstance(gate, WienerGate)
        self.assertFalse(hasattr(gate, 'k_rel'))
        self.assertFalse(hasattr(gate, 'bogus'))

    def test_relative_ceiling_dispatch(self):
        gate = build_confidence_gate('relative_ceiling', xp=np, n_subaps=4, dtype=np.float64,
                                      k_wiener=10.0, gate_params={'k_rel': 0.3, 'margin_thresh': 0.5})
        self.assertIsInstance(gate, RelativeCeilingGate)
        self.assertEqual(gate.k_rel, 0.3)
        self.assertEqual(gate.margin_thresh, 0.5)
        np.testing.assert_array_equal(gate.rho_sq_ceiling, np.full(4, 10.0))

    def test_shifted_sigmoid_dispatch(self):
        gate = build_confidence_gate('shifted_sigmoid', xp=np, n_subaps=4, dtype=np.float64,
                                      k_wiener=10.0,
                                      gate_params={'boost_mult': 6.0, 'beta_snr': 0.5,
                                                   'margin_thresh': 0.25, 'beta_margin': 2.0})
        self.assertIsInstance(gate, ShiftedSigmoidGate)
        self.assertEqual(gate.rho_boost_centre, 60.0)
        self.assertEqual(gate.margin_thresh, 0.25)
        self.assertEqual(gate.beta_margin, 2.0)

    def test_none_gate_params_is_equivalent_to_empty_dict(self):
        gate_none = build_confidence_gate('relative_ceiling', xp=np, n_subaps=2,
                                           dtype=np.float64, k_wiener=10.0, gate_params=None)
        gate_empty = build_confidence_gate('relative_ceiling', xp=np, n_subaps=2,
                                            dtype=np.float64, k_wiener=10.0, gate_params={})
        self.assertEqual(gate_none.k_rel, gate_empty.k_rel)
        self.assertEqual(gate_none.margin_thresh, gate_empty.margin_thresh)

    def test_unknown_gate_type_raises_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            build_confidence_gate('not_a_real_gate', xp=np, n_subaps=4, dtype=np.float64,
                                   k_wiener=10.0)
        message = str(ctx.exception)
        self.assertIn('not_a_real_gate', message)
        self.assertIn('wiener', message)
        self.assertIn('relative_ceiling', message)
        self.assertIn('shifted_sigmoid', message)


class TestConfidenceGateBase(unittest.TestCase):
    """Base class contract: default needs_margin/telemetry(), compute()
    left to subclasses."""

    def test_default_needs_margin_is_false(self):
        self.assertFalse(ConfidenceGate.needs_margin)

    def test_default_telemetry_is_empty(self):
        self.assertEqual(ConfidenceGate().telemetry(), {})

    def test_default_compute_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            ConfidenceGate().compute(np, np.array([1.0]), None, 1e-12)


if __name__ == '__main__':
    unittest.main()
