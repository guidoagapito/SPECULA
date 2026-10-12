import unittest
import os
import glob
import shutil
from astropy.io import fits

import specula
specula.init(0)

from specula.lib.demodulate_signal import demodulate_signal
from test.specula_testlib import cpu_and_gpu
from specula.simul import Simul
from specula import np, cpuArray


def _demodulate_signal_loop_reference(signal_data, carrier_freq, sampling_freq,
                                      xp=np, dtype=np.float32):
    """Previous O(nt^2) implementation, kept as a reference for the cumsum version."""
    data = xp.asarray(signal_data, dtype=dtype)
    is_1d = (data.ndim == 1)
    if is_1d:
        data = data[:, xp.newaxis]
    nt, nsignals = data.shape

    dt = 1.0 / sampling_freq
    t = xp.arange(nt, dtype=dtype) * dt
    w = 2 * xp.pi * carrier_freq

    periods = int(xp.floor(xp.max(t) * carrier_freq))
    if periods > 0:
        test_vect = (xp.arange(periods) + 1) * sampling_freq / carrier_freq
        errors = xp.abs(test_vect - xp.round(test_vect))
        idx = xp.where(errors <= 1e-3)[0]
        if len(idx) > 0:
            n4mean = int(test_vect[xp.max(idx)])
        else:
            n4mean = int(test_vect[xp.argmin(errors)])
    else:
        n4mean = max(1, nt // 4)

    data_mean = xp.mean(data, axis=0, keepdims=True)
    cur_data = data - data_mean
    tilt = (cur_data[-1:, :] - cur_data[0:1, :]) / nt
    t_ramp = xp.arange(nt, dtype=dtype)[:, xp.newaxis]
    cur_data = cur_data - tilt * t_ramp - cur_data[0:1, :]

    sin_carrier = xp.sin(w * t)[:, xp.newaxis]
    cos_carrier = xp.cos(w * t)[:, xp.newaxis]
    qa_ref = xp.mean(cur_data * sin_carrier, axis=0)
    pa_ref = xp.mean(cur_data * cos_carrier, axis=0)
    pphi0 = xp.arctan2(qa_ref, pa_ref)

    dem_sin = xp.sin(w * t[:, xp.newaxis] - pphi0[xp.newaxis, :])
    dem_cos = xp.cos(w * t[:, xp.newaxis] - pphi0[xp.newaxis, :])

    qa = xp.zeros((nt, nsignals), dtype=dtype)
    pa = xp.zeros((nt, nsignals), dtype=dtype)
    for j in range(2, nt):
        window_data = data[:j+1, :] - xp.mean(data[:j+1, :], axis=0, keepdims=True)
        window_tilt = (window_data[j:j+1, :] - window_data[0:1, :]) / j
        t_window = xp.arange(j+1, dtype=dtype)[:, xp.newaxis]
        window_data = window_data - window_tilt * t_window - window_data[0:1, :]
        qa[j, :] = xp.sum(window_data * dem_sin[:j+1, :], axis=0) / (j + 1)
        pa[j, :] = xp.sum(window_data * dem_cos[:j+1, :], axis=0) / (j + 1)

    data_dem_temp = 2.0 * xp.sqrt(qa[2:, :]**2 + pa[2:, :]**2)
    pphi_temp = xp.arctan2(qa[2:, :], pa[2:, :])

    start_idx = max(0, nt - 2 - n4mean)
    end_idx = nt - 2
    if end_idx > start_idx:
        value = xp.mean(data_dem_temp[start_idx:end_idx, :], axis=0)
        pphi = xp.mean(pphi_temp[start_idx:end_idx, :], axis=0)
    else:
        value = data_dem_temp[-1, :] if len(data_dem_temp) > 0 else xp.zeros(nsignals)
        pphi = pphi_temp[-1, :] if len(pphi_temp) > 0 else xp.zeros(nsignals)

    pphi = pphi + pphi0
    if is_1d:
        return float(value[0]), float(pphi[0])
    return value, pphi


def _wrap_phase(delta):
    """Wrap a phase difference to [-pi, pi)."""
    return (delta + np.pi) % (2 * np.pi) - np.pi


def _make_sprint_like_signals(nt, nsignals=16, sampling_freq=1000.0, noise_std=0.5, seed=1234):
    """
    Host-side (numpy) SPRINT-like multi-signal input: per-signal sinusoid at
    sampling_freq/35 Hz with random amplitude and phase, DC offset, linear drift
    and Gaussian noise. Returns (signals, carrier_freq, amplitudes, phases).
    """
    rng = np.random.default_rng(seed)
    carrier_freq = sampling_freq / 35
    t = np.arange(nt) / sampling_freq
    amplitudes = rng.uniform(0.5, 5.0, nsignals)
    phases = rng.uniform(-np.pi, np.pi, nsignals)
    drift = rng.uniform(-5.0, 5.0, nsignals)  # units/s
    signals = (amplitudes * np.sin(2 * np.pi * carrier_freq * t[:, None] + phases)
               + 50.0 + drift * t[:, None]
               + rng.normal(0.0, noise_std, (nt, nsignals)))
    return signals, carrier_freq, amplitudes, phases

class TestDemodulator(unittest.TestCase):
    """Test demodulator by running a simulation and checking the output amplitude"""

    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.params_file = os.path.join(os.path.dirname(__file__), 'params_demodulator_test.yml')
        os.makedirs(self.datadir, exist_ok=True)
        self.expected_amplitude = 5.0  # From params_demodulator_test.yml

        # Get current working directory
        self.cwd = os.getcwd()

    def tearDown(self):
        # Remove test/data directory with timestamp
        data_dirs = glob.glob(os.path.join(self.datadir, '2*'))
        for data_dir in data_dirs:
            if os.path.isdir(data_dir):
                try:
                    shutil.rmtree(data_dir)
                except Exception:
                    pass

        # Change back to original directory
        os.chdir(self.cwd)

    def test_demodulator_amplitude(self):
        """Run the simulation and check demodulator output amplitude"""

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Run the simulation
        simul = Simul(self.params_file)
        simul.run()

        # Find the most recent data directory (with timestamp)
        data_dirs = sorted(glob.glob(os.path.join(self.datadir, '2*')))
        self.assertTrue(data_dirs, "No data directory found after simulation")
        latest_data_dir = data_dirs[-1]

        # Check if demodulator output file exists
        demod_file = os.path.join(latest_data_dir, 'dem.fits')
        self.assertTrue(os.path.exists(demod_file),
                        f"Demodulator output file not found: {demod_file}")

        # Read demodulator output
        with fits.open(demod_file) as hdul:
            self.assertTrue(len(hdul) >= 1, "No data found in demodulator output file")
            demod_values = hdul[0].data.copy()
            self.assertIsNotNone(demod_values,
                                 "No data found in first HDU of demodulator output file")

            # Check that the output matches the input amplitude (within tolerance)
            mean_demod = np.mean(demod_values)
            tolerance = 0.05 * self.expected_amplitude  # 5% tolerance
            self.assertTrue(
                abs(mean_demod - self.expected_amplitude) < tolerance,
                f"Demodulator output {mean_demod:.3f} does not match expected"
                f" amplitude {self.expected_amplitude:.3f} (tol={tolerance:.3f})"
            )
            print(f"Demodulator output OK: mean={mean_demod:.3f},"
                  f" expected={self.expected_amplitude:.3f}")

class TestDemodulateSignal(unittest.TestCase):
    """Test suite for demodulate_signal function"""

    verbose = False  # Set to True to print debug info during tests

    @cpu_and_gpu
    def test_demodulate_single_signal_no_noise(self, target_device_idx, xp):
        """Test demodulation of a single sinusoidal signal without noise"""

        if self.verbose: # pragma: no cover
            print(f"\n{'='*70}")
            print(f"Testing single signal demodulation (no noise)")
            print(f"  target_device={target_device_idx}")
            print(f"{'='*70}")

        # Signal parameters
        duration = 0.2  # seconds
        dt = 0.001  # 1ms -> 1000 Hz sampling
        sampling_freq = 1.0 / dt
        carrier_freq = 10.0  # Hz
        amplitude_true = 2.5
        phase_true = 0.0  # Start at phase 0 (sine wave)

        # Generate time vector
        time = xp.arange(0, duration, dt)
        nt = len(time)

        # Generate pure sinusoidal signal
        signal = amplitude_true * xp.sin(2 * xp.pi * carrier_freq * time + phase_true)

        if self.verbose: # pragma: no cover
            print(f"\nSignal parameters:")
            print(f"  Duration: {duration}s")
            print(f"  Sampling freq: {sampling_freq} Hz")
            print(f"  Carrier freq: {carrier_freq} Hz")
            print(f"  True amplitude: {amplitude_true}")
            print(f"  Number of samples: {nt}")
            print(f"  Number of periods: {carrier_freq * duration}")

        # Demodulate
        amp_demod, phase_demod = demodulate_signal(
            signal, carrier_freq, sampling_freq,
            cumulated=True, xp=xp
        )

        # Apply phase correction to get signed amplitude
        amp_signed = float(amp_demod * xp.sign(xp.cos(phase_demod)))
        amp_demod = float(amp_demod)
        phase_demod = float(phase_demod)

        if self.verbose: # pragma: no cover
            print(f"\nDemodulation results:")
            print(f"  Demodulated amplitude: {amp_demod:.6f}")
            print(f"  Demodulated phase: {phase_demod:.6f} rad ({np.degrees(phase_demod):.2f} deg)")
            print(f"  Signed amplitude: {amp_signed:.6f}")
            print(f"  Amplitude error: {abs(amp_signed - amplitude_true):.6e}")
            print(f"  Relative error:"
                  f" {abs(amp_signed - amplitude_true) / amplitude_true * 100:.3f}%")

        # Check results (should be very accurate)
        self.assertIsInstance(amp_demod, float, "Amplitude should be scalar")
        self.assertIsInstance(phase_demod, float, "Phase should be scalar")
        self.assertAlmostEqual(amp_demod, amplitude_true, delta=0.1,
             msg=f"Amplitude error too large: {abs(amp_demod - amplitude_true):.3e}")

    @cpu_and_gpu
    def test_demodulate_single_signal_with_noise(self, target_device_idx, xp):
        """Test demodulation of a single sinusoidal signal with noise"""

        if self.verbose: # pragma: no cover
            print(f"\n{'='*70}")
            print(f"Testing single signal demodulation (with noise)")
            print(f"  target_device={target_device_idx}")
            print(f"{'='*70}")

        # Signal parameters
        duration = 1.0  # Longer for better SNR
        dt = 0.001
        sampling_freq = 1.0 / dt
        carrier_freq = 15.0  # Hz
        amplitude_true = 5.0
        noise_std = 0.5  # 10% noise

        # Generate time vector
        time = xp.arange(0, duration, dt)

        # Generate noisy signal
        signal_clean = amplitude_true * xp.sin(2 * xp.pi * carrier_freq * time)
        noise = xp.random.normal(0, noise_std, len(time))
        signal_noisy = signal_clean + noise

        if self.verbose: # pragma: no cover
            print(f"\nSignal parameters:")
            print(f"  True amplitude: {amplitude_true}")
            print(f"  Noise std: {noise_std}")
            print(f"  SNR: {amplitude_true / noise_std:.1f}")

        # Demodulate
        amp_demod, phase_demod = demodulate_signal(
            signal_noisy, carrier_freq, sampling_freq,
            cumulated=True, xp=xp
        )

        amp_signed = cpuArray(amp_demod * xp.sign(xp.cos(phase_demod)))
        amp_demod = cpuArray(amp_demod)
        phase_demod = cpuArray(phase_demod)

        if self.verbose: # pragma: no cover
            print(f"\nDemodulation results:")
            print(f"  Demodulated amplitude: {amp_demod:.6f}")
            print(f"  Signed amplitude: {amp_signed:.6f}")
            print(f"  Amplitude error: {abs(amp_signed - amplitude_true):.6e}")
            print(f"  Relative error: "
                  f" {abs(amp_signed - amplitude_true) / amplitude_true * 100:.3f}%")

        # With noise, allow larger tolerance
        self.assertAlmostEqual(amp_demod, amplitude_true, delta=0.5,
             msg=f"Amplitude error too large with noise: {abs(amp_demod - amplitude_true):.3e}")

    @cpu_and_gpu
    def test_demodulate_multiple_signals_vectorized(self, target_device_idx, xp):
        """Test vectorized demodulation of multiple signals simultaneously"""

        if self.verbose: # pragma: no cover
            print(f"\n{'='*70}")
            print(f"Testing vectorized demodulation (multiple signals)")
            print(f"  target_device={target_device_idx}")
            print(f"{'='*70}")

        # Signal parameters
        duration = 0.2
        dt = 0.001
        sampling_freq = 1.0 / dt
        carrier_freq = 20.0  # Hz

        # Generate different amplitudes for different "slopes"
        nsignals = 10
        amplitudes_true = xp.linspace(1.0, 5.0, nsignals)

        # Generate time vector
        time = xp.arange(0, duration, dt)
        nt = len(time)

        # Generate 2D signal array: (nt, nsignals)
        signals_2d = xp.zeros((nt, nsignals))
        for i in range(nsignals):
            # Each signal has different amplitude, same frequency
            signals_2d[:, i] = amplitudes_true[i] * xp.sin(2 * xp.pi * carrier_freq * time)

        if self.verbose: # pragma: no cover
            print(f"\nSignal parameters:")
            print(f"  Number of signals: {nsignals}")
            print(f"  Carrier freq: {carrier_freq} Hz")
            print(f"  True amplitudes range: [{float(amplitudes_true.min()):.2f},"
                f" {float(amplitudes_true.max()):.2f}]")
            print(f"  Signal shape: {signals_2d.shape}")

        # Demodulate all signals at once (vectorized)
        amps_demod, phases_demod = demodulate_signal(
            signals_2d, carrier_freq, sampling_freq,
            cumulated=True, xp=xp
        )

        if self.verbose: # pragma: no cover
            print(f"\nDemodulation results:")
            print(f"  Output amplitudes shape: {amps_demod.shape}")
            print(f"  Output phases shape: {phases_demod.shape}")
            print(f"  Demodulated amplitudes range:"
                  f" [{float(amps_demod.min()):.2f}, {float(amps_demod.max()):.2f}]")

        # Check shapes
        self.assertEqual(amps_demod.shape, (nsignals,), "Amplitudes should be 1D array")
        self.assertEqual(phases_demod.shape, (nsignals,), "Phases should be 1D array")

        sign = None
        # Check each amplitude
        for i in range(nsignals):
            amp_signed = amps_demod[i] * xp.sign(xp.cos(phases_demod[i]))
            # find correct sign at i == 0
            if sign is None:
                sign = xp.sign(amp_signed * amplitudes_true[i])
            amp_signed *= sign
            error_rel = abs(amp_signed - amplitudes_true[i]) / amplitudes_true[i]

            if (i == 0 or i == nsignals - 1) and self.verbose:  # pragma: no cover
                print(f"  Signal {i}: true={float(amplitudes_true[i]):.3f}, "
                      f"demod={float(amp_signed):.3f}, error={float(error_rel*100):.2f}%")

            self.assertLess(error_rel, 0.05,
                           f"Signal {i} error too large: {float(error_rel*100):.1f}%")

        if self.verbose: # pragma: no cover
            print(f"  All {nsignals} signals demodulated successfully!")

    @cpu_and_gpu
    def test_demodulate_multiple_signals_different_phases(self, target_device_idx, xp):
        """Test vectorized demodulation with different phases"""

        if self.verbose: # pragma: no cover
            print(f"\n{'='*70}")
            print(f"Testing vectorized demodulation (different phases)")
            print(f"  target_device={target_device_idx}")
            print(f"{'='*70}")

        # Signal parameters
        duration = 0.5
        dt = 0.001
        sampling_freq = 1.0 / dt
        carrier_freq = 20.0
        amplitude = 3.0
        nsignals = 3

        # Different phases
        phases_true = xp.linspace(0, xp.pi, nsignals)

        # Generate time vector
        time = xp.arange(0, duration, dt)
        nt = len(time)

        # Generate signals with different phases
        signals_2d = xp.zeros((nt, nsignals))
        for i in range(nsignals):
            signals_2d[:, i] = amplitude * xp.sin(2 * xp.pi * carrier_freq * time + phases_true[i])

        if self.verbose: # pragma: no cover
            print(f"\nSignal parameters:")
            print(f"  Number of signals: {nsignals}")
            print(f"  True amplitude: {amplitude}")
            print(f"  True phases: {float(xp.degrees(phases_true))}")

        # Demodulate
        amps_demod, phases_demod = demodulate_signal(
            signals_2d, carrier_freq, sampling_freq,
            cumulated=True, xp=xp
        )

        amps_demod = cpuArray(amps_demod)

        if self.verbose: # pragma: no cover
            # All should have similar amplitude
            print(f"\nDemodulation results:")
            print(f"  Demodulated amplitudes: {amps_demod}")
            print(f"  Mean amplitude: {amps_demod.mean():.3f} ± {amps_demod.std():.3f}")

        # Check that all amplitudes are close to true value
        for i in range(nsignals):
            self.assertAlmostEqual(amps_demod[i], amplitude, delta=0.2,
                                  msg=f"Signal {i} amplitude error too large")

    @cpu_and_gpu
    def test_demodulate_sprint_like_scenario(self, target_device_idx, xp):
        """Test demodulation in a SPRINT-like scenario"""

        if self.verbose: # pragma: no cover
            print(f"\n{'='*70}")
            print(f"Testing SPRINT-like demodulation scenario")
            print(f"  target_device={target_device_idx}")
            print(f"{'='*70}")

        # Simulate SPRINT scenario:
        # - Multiple subapertures (slopes)
        # - Single mode with known amplitude
        # - Short time series

        duration = 0.1  # 100ms
        dt = 0.001
        sampling_freq = 1.0 / dt
        carrier_freq = 50.0  # Higher frequency

        # Simulate interaction matrix for one mode
        n_subaps = 10
        nslopes = 2 * n_subaps  # X and Y slopes

        # Random IM coefficients (positive and negative)
        im_mode = xp.random.randn(nslopes) * 2.0

        if self.verbose: # pragma: no cover
            print(f"\nSPRINT scenario:")
            print(f"  Number of subapertures: {n_subaps}")
            print(f"  Number of slopes: {nslopes}")
            print(f"  Carrier frequency: {carrier_freq} Hz")
            print(f"  Duration: {duration}s ({carrier_freq * duration:.0f} periods)")
            print(f"  IM RMS: {float(xp.sqrt(xp.mean(im_mode**2))):.3e}")

        # Generate time vector
        time = xp.arange(0, duration, dt)
        nt = len(time)

        # Generate modulated slopes (each slope is IM coefficient * sine)
        modulation = xp.sin(2 * xp.pi * carrier_freq * time)
        slopes_time = xp.outer(modulation, im_mode)  # Shape: (nt, nslopes)

        if self.verbose: # pragma: no cover
            print(f"  Slopes time series shape: {slopes_time.shape}")
            print(f"  Slopes RMS: {float(xp.sqrt(xp.mean(slopes_time**2))):.3e}")

        # Demodulate (vectorized)
        amps_demod, phases_demod = demodulate_signal(
            slopes_time, carrier_freq, sampling_freq,
            cumulated=True, xp=xp
        )

        # Apply phase correction
        im_reconstructed = xp.array(amps_demod) * xp.sign(xp.cos(xp.array(phases_demod)))

        if self.verbose: # pragma: no cover
            print(f"\nReconstruction quality:")
            print(f"  Demodulated IM shape: {im_reconstructed.shape}")
            print(f"  Demodulated IM RMS: {float(xp.sqrt(xp.mean(im_reconstructed**2))):.3e}")

        # Compare with true IM
        # find correct global sign
        sign = xp.sign(xp.dot(im_reconstructed, im_mode))
        im_reconstructed *= sign
        im_diff = im_reconstructed - im_mode
        rms_error = xp.sqrt(xp.mean(im_diff**2))
        rms_ref = xp.sqrt(xp.mean(im_mode**2))
        rel_error = rms_error / rms_ref

        plot_debug = False
        if plot_debug: # pragma: no cover
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(cpuArray(im_mode), label='True IM')
            plt.plot(cpuArray(im_reconstructed), label='Demodulated IM', linestyle='--')
            plt.legend()
            plt.title('IM Reconstruction')
            plt.subplot(1, 2, 2)
            plt.plot(cpuArray(im_diff))
            plt.title('IM Reconstruction Error')
            plt.tight_layout()
            plt.show()

        if self.verbose: # pragma: no cover
            print(f"  Reconstruction error RMS: {float(rms_error):.3e}")
            print(f"  Relative error: {float(rel_error) * 100:.2f}%")

        # Should reconstruct IM accurately
        self.assertLess(float(rel_error), 0.05,
                       f"IM reconstruction error too large: {float(rel_error)*100:.1f}%")

        if self.verbose: # pragma: no cover
            # Check individual slopes (sample a few)
            print(f"\nSample slope reconstruction:")
            for i in [0, nslopes//4, nslopes//2, 3*nslopes//4, nslopes-1]:
                error = abs(im_reconstructed[i] - im_mode[i])
                print(f"  Slope {i:3d}: true={float(im_mode[i]):7.3f}, "
                    f"recon={float(im_reconstructed[i]):7.3f}, error={float(error):.3e}")

    def test_demodulate_edge_cases(self):
        """Test edge cases and error handling"""

        if self.verbose: # pragma: no cover
            print(f"\n{'='*70}")
            print(f"Testing edge cases")
            print(f"{'='*70}")

        # Very short signal
        short_signal = np.sin(2 * np.pi * 5.0 * np.arange(10) * 0.001)
        amp, phase = demodulate_signal(short_signal, 5.0, 1000.0, xp=np)
        self.assertIsInstance(amp, float)
        if self.verbose: # pragma: no cover
            print(f"  Short signal (10 samples): OK")

        # Single sample (should not crash)
        try:
            single = np.array([1.0])
            amp, phase = demodulate_signal(single, 5.0, 1000.0, xp=np)
            if self.verbose: # pragma: no cover
                print(f"  Single sample: OK (amp={amp:.3f})")
        except Exception as e:
            if self.verbose: # pragma: no cover
                print(f"  Single sample: Expected exception: {e}")

        # Zero signal
        zero_signal = np.zeros(100)
        amp, phase = demodulate_signal(zero_signal, 5.0, 1000.0, xp=np)
        self.assertAlmostEqual(amp, 0.0, delta=1e-6)
        if self.verbose: # pragma: no cover
            print(f"  Zero signal: OK (amp={amp:.3e})")

        # Very high frequency (Nyquist limit)
        nyquist_freq = 500.0  # fs=1000, Nyquist=500
        time = np.arange(0, 1.0, 0.001)
        signal_nyquist = np.sin(2 * np.pi * nyquist_freq * time)
        amp, phase = demodulate_signal(signal_nyquist, nyquist_freq, 1000.0, xp=np)
        if self.verbose: # pragma: no cover
            print(f"  Nyquist frequency: amp={amp:.3f}")

    def test_demodulate_cumulated_vs_simple(self):
        """Compare cumulated and simple demodulation methods"""

        if self.verbose: # pragma: no cover
            print(f"\n{'='*70}")
            print(f"Testing cumulated vs simple demodulation")
            print(f"{'='*70}")

        # Generate test signal
        duration = 0.2
        dt = 0.001
        carrier_freq = 10.0
        amplitude = 3.0
        time = np.arange(0, duration, dt)
        signal = amplitude * np.sin(2 * np.pi * carrier_freq * time)

        # Cumulated demodulation
        amp_cum, phase_cum = demodulate_signal(
            signal, carrier_freq, 1.0/dt, cumulated=True, xp=np
        )

        # Simple demodulation
        amp_sim, phase_sim = demodulate_signal(
            signal, carrier_freq, 1.0/dt, cumulated=False, xp=np
        )

        if self.verbose: # pragma: no cover
            print(f"\nComparison:")
            print(f"  True amplitude: {amplitude}")
            print(f"  Cumulated: amp={amp_cum:.6f}, phase={phase_cum:.6f}")
            print(f"  Simple:    amp={amp_sim:.6f}, phase={phase_sim:.6f}")
            print(f"  Difference: amp={abs(amp_cum - amp_sim):.6e}, "
                f"phase={abs(phase_cum - phase_sim):.6e}")

        # Both should give similar results for clean signal
        self.assertAlmostEqual(amp_cum, amp_sim, delta=0.1)

    def test_demodulate_cumulated_dtype_follows_argument(self):
        """
        With cumulated=True, the internal qa/pa arrays (and thus the output) must
        follow the requested `dtype`, not be hardcoded to float32.
        """
        duration = 0.05
        dt = 0.001
        sampling_freq = 1.0 / dt
        carrier_freq = 10.0

        time = np.arange(0, duration, dt)
        nsignals = 2
        signals_2d = np.zeros((len(time), nsignals))
        for i in range(nsignals):
            signals_2d[:, i] = (i + 1) * np.sin(2 * np.pi * carrier_freq * time)

        amp64, phase64 = demodulate_signal(signals_2d, carrier_freq, sampling_freq,
                                          cumulated=True, xp=np, dtype=np.float64)
        self.assertEqual(amp64.dtype, np.float64)
        self.assertEqual(phase64.dtype, np.float64)

        amp32, phase32 = demodulate_signal(signals_2d, carrier_freq, sampling_freq,
                                          cumulated=True, xp=np, dtype=np.float32)
        self.assertEqual(amp32.dtype, np.float32)
        self.assertEqual(phase32.dtype, np.float32)

    def test_demodulate_multiple_modes(self):
        """Demodulate 2d vectors"""

        if self.verbose: # pragma: no cover
            print(f"\n{'='*70}")
            print(f"Testing demodulation of multiple signals")
            print(f"{'='*70}")

        # Generate test signals
        duration = 0.2
        dt = 0.001
        carrier_freq = 10.0
        amplitude = 3.0
        time = np.arange(0, duration, dt)
        phase_shift = np.pi/3
        signal1 = amplitude * np.sin(2 * np.pi * carrier_freq * time)
        signal2 = amplitude * 0.5 * np.sin(2 * np.pi * carrier_freq * time + phase_shift)

        signal = np.array([signal1, signal2]).T

        # Simple demodulation
        amp, phase = demodulate_signal(
            signal, carrier_freq, 1.0/dt, cumulated=False, xp=np
        )

        diff_amp = amp[0] / amp[1]
        diff_phase = phase[0] - phase[1]

        self.assertAlmostEqual(diff_amp, 2.0, places=1)
        self.assertAlmostEqual(diff_phase, np.pi/3, places=1)


class TestDemodulateSignalCumsum(unittest.TestCase):
    """Cumsum implementation of the cumulated demodulation vs the O(nt^2) loop."""

    sampling_freq = 1000.0

    def _compare(self, xp, nt, dtype, rtol_amp, atol_phase):
        signals, fc, _, _ = _make_sprint_like_signals(nt)
        sig = xp.asarray(signals)
        amp_new, ph_new = demodulate_signal(sig, fc, self.sampling_freq,
                                            cumulated=True, xp=xp, dtype=dtype)
        amp_ref, ph_ref = _demodulate_signal_loop_reference(sig, fc, self.sampling_freq,
                                                            xp=xp, dtype=dtype)
        amp_new, ph_new = cpuArray(amp_new), cpuArray(ph_new)
        amp_ref, ph_ref = cpuArray(amp_ref), cpuArray(ph_ref)
        self.assertEqual(amp_new.dtype, np.dtype(dtype))
        self.assertEqual(ph_new.dtype, np.dtype(dtype))
        np.testing.assert_allclose(amp_new, amp_ref, rtol=rtol_amp, atol=0)
        np.testing.assert_allclose(_wrap_phase(ph_new - ph_ref), 0.0, atol=atol_phase)

    @cpu_and_gpu
    def test_cumsum_matches_loop_float64(self, target_device_idx, xp):
        """Both are exact reformulations of the same sums: float64 agrees to rounding."""
        # Measured (CPU, nt=2000, 16 signals): amplitude rel. diff ~4e-15, phase ~2e-15 rad.
        # 1e-9 leaves ~5 orders of magnitude of margin (and covers GPU summation order).
        self._compare(xp, 2000, np.float64, rtol_amp=1e-9, atol_phase=1e-9)

    @cpu_and_gpu
    def test_cumsum_matches_loop_float32(self, target_device_idx, xp):
        """In float32 the loop is limited by float32 rounding; new one accumulates in float64."""
        # Measured (CPU, nt=2000): amplitude rel. diff ~2.6e-7, phase diff 0.
        # 1e-4 is ~400x above that, still ~1e4 times tighter than the physical errors.
        self._compare(xp, 2000, np.float32, rtol_amp=1e-4, atol_phase=1e-4)

    @cpu_and_gpu
    def test_cumsum_matches_loop_short_inputs(self, target_device_idx, xp):
        """Short inputs exercise the j >= 2 start and the n4mean fallback branches."""
        # Measured float64: up to ~5e-13 (nt=3, ill-conditioned), float32: up to ~2.4e-7.
        for nt in (3, 4, 10):
            with self.subTest(nt=nt, dtype='float64'):
                self._compare(xp, nt, np.float64, rtol_amp=1e-9, atol_phase=1e-9)
            with self.subTest(nt=nt, dtype='float32'):
                self._compare(xp, nt, np.float32, rtol_amp=1e-4, atol_phase=1e-4)

    @cpu_and_gpu
    def test_cumsum_known_sinusoid_amplitude_and_phase(self, target_device_idx, xp):
        """
        Noise-free A sin(2 pi f t + phi) + DC + linear drift, 5 s at 1 kHz.
        Expected amplitude A; phase follows the module convention: the reference
        phase is arctan2(<s sin>, <s cos>) = pi/2 - phi (mod 2 pi).
        """
        fs = self.sampling_freq
        fc = fs / 35
        t = np.arange(5000) / fs
        amplitudes = np.array([1.0, 2.5, 4.0])
        phases = np.array([-2.0, 0.3, 1.2])
        drift = np.array([1.0, -3.0, 2.0])
        signals = (amplitudes * np.sin(2 * np.pi * fc * t[:, None] + phases)
                   + 50.0 + drift * t[:, None])

        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype.__name__):
                amp, ph = demodulate_signal(xp.asarray(signals), fc, fs,
                                            cumulated=True, xp=xp, dtype=dtype)
                self.assertEqual(amp.dtype, dtype)
                self.assertEqual(ph.dtype, dtype)
                amp, ph = cpuArray(amp), cpuArray(ph)
                # Measured: amplitude error <= 1.4e-2 (0.35% of A), phase error <= 2.5e-3 rad
                # (residual from the progressive-window detrending of a drifting signal).
                np.testing.assert_allclose(amp, amplitudes, rtol=1e-2)
                np.testing.assert_allclose(_wrap_phase(ph - (np.pi / 2 - phases)),
                                           0.0, atol=1e-2)

    @cpu_and_gpu
    def test_cumsum_known_single_signal(self, target_device_idx, xp):
        """1D input returns Python floats close to the injected amplitude."""
        fs = self.sampling_freq
        fc = fs / 35
        t = np.arange(5000) / fs
        sig = 3.0 * np.sin(2 * np.pi * fc * t + 0.7) + 50.0 + 2.0 * t
        amp, ph = demodulate_signal(xp.asarray(sig), fc, fs, cumulated=True,
                                    xp=xp, dtype=np.float64)
        self.assertIsInstance(amp, float)
        self.assertIsInstance(ph, float)
        self.assertAlmostEqual(amp, 3.0, delta=0.03)
        self.assertAlmostEqual(float(_wrap_phase(ph - (np.pi / 2 - 0.7))), 0.0, delta=1e-2)
