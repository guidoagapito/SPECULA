from collections import deque

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
    Step-1 supervisor for spot (re)acquisition of a single-subaperture WCoG tip/tilt sensor (E-MAO-PC0-INA-TNO-009).

    Looks: each look is the mean of ``look_frames`` frames registered (Fourier shift) with the applied tip/tilt
    command, correlated over the whole frame with a core + halo matched filter; its peak, registered with the
    command (disturbance coordinates), is one consensus estimate, so that the spot motion caused by our own
    correction does not spoil the consensus. When the last ``n_cons`` estimates agree (within ``delta_factor`` *
    sigma_w of their median), the latest one lies farther than that from the WCoG window, and guard C passes (the
    correlation peak within ``r_loc_sigma`` sigma_w of the window is below ``q_thr`` of the global peak), the spot
    is lost and the window (the slope zero of :class:`ShSlopecMovable`) is put on the estimated spot position: the
    loop stabilises the spot where it is. The move is then confirmed: a block of ``confirm_frames`` frames
    (default ``block_frames``) at the new window must show local z > ``z_thr_local`` or window flux > ``flux_thr``,
    otherwise the move is aborted and the window restored. After the confirmation, and at least ``k_hold`` frames
    after the move, a feedforward tip/tilt command equal to the window offset is issued and the window returns to
    the reference when the feedforward reaches the pixels.

    Presence (``presence``, default on): blocks of ``block_frames`` frames, not registered (the tracked spot is
    stationary on the detector). A block is present if ANY of the global z (peak - mean) / std of the correlation
    map > ``z_thr``, the local z (same normalisation, peak within ``r_loc_sigma`` sigma_w of the window) >
    ``z_thr_local``, or the window flux (Gaussian WCoG mask on the window, per-frame clip at 0) > ``flux_thr``. A
    dropout is declared after ``k_absent`` absent blocks: the hold flag of ``out_window`` is set
    (:class:`ShSlopecMovable` then outputs zero slopes), no look is taken and nothing is moved until ``k_present``
    present blocks. The thresholds are noise-only quantiles calibrated with the presence template.

    ``presence_tpl_fwhm`` / ``presence_halo_fwhm`` / ``presence_halo_fraction`` (default: the look template) give
    presence and confirmation their own core + halo template, e.g. fitted to the measured long-exposure PSF: for a
    broad PSF the fixed template makes the z statistics miss a present star, while a halo-dominated template used for
    the looks would smooth the correlation map (imprecise peaks, guard C vetoing lost spots).

    ``ref_offset`` is the offset of the matched-filter peak from the WCoG centroid for a spot at the
    reference (peak/centroid asymmetry, integer-pixel quantisation): the estimates are the centroid
    displacement from the reference, so the window centre ``w`` needs no further correction.

    Coordinates: x = column, y = row, in pixels from the subaperture centre. ``cmd_to_px`` maps the
    tip/tilt command (nm) to the spot shift produced by the correction (px per nm, sign and axes to be
    calibrated on the system).

    ``cmd_latency`` = L (frames, required): ``in_command`` is the command as issued by the controller, which,
    like the feedforward, reaches the pixels L frames after the step it is read/issued. Frames are registered
    with the command L frames old, and the window returns only when the feedforward is expected on the pixels;
    no new move starts while a return is in flight.

    Outputs: ``out_window`` = [wx, wy, hold] (feed it to ShSlopecMovable, delayed by one step),
    ``out_feedforward`` = accumulated feedforward command (nm, feed it to an extra tip/tilt DM),
    ``out_state`` = telemetry [est_x, est_y, z, guard_ratio, dropout, n_moves, z_local, window_flux, n_aborts]
    (z, z_local, window_flux: last completed presence block, NaN if presence is off),
    ``out_look`` = image of the last complete look (mean of ``look_frames`` frames registered to the command of the
    first one, i.e. in the detector frame of that first frame; the frame itself with ``look_frames`` = 1), held
    until the next look is complete. Optional input ``in_enable`` (> 0.5 = enabled, default enabled when not
    connected): while disabled the supervisor only registers the frames and forms the looks (``out_look`` keeps
    updating), with no presence logic, consensus, move or feedforward; on re-enabling it restarts from its initial
    state (window at the reference, no feedforward). Used to keep the supervisor passive in open loop; with
    ``hold_when_disabled`` the hold flag is also set while disabled, so the loop is open (zero slopes).
    The supervisor works on detector frames: it processes a step only if ``in_pixels`` was refreshed in it (an
    input refreshed at every simulation step, such as ``in_enable`` from a generator, would otherwise make it
    count a frame of a longer detector integration more than once), and only then refreshes its outputs.

    Earlier options (return modes 'one' and 'n3', a restricted search radius, registered presence blocks, the
    legacy command semantics cmd_latency = None, global-z-only presence, moves without confirmation) are in the
    git tag ``spot-supervisor-legacy``, needed to reproduce the campaigns that used them.
    """

    def __init__(self,
                 weighted_pix_rad: float,
                 cmd_latency: int,
                 np_sub: int = 240,
                 cmd_to_px: list = None,
                 ref_offset: list = None,
                 n_cons: int = 2,
                 delta_factor: float = 1.5,
                 q_thr: float = 0.6,
                 r_loc_sigma: float = 3.0,
                 k_hold: int = None,
                 tpl_fwhm: float = 2.0,
                 halo_fwhm: float = 15.0,
                 halo_fraction: float = 0.3,
                 presence: bool = True,
                 z_thr: float = None,
                 block_frames: int = 1,
                 k_absent: int = 8,
                 k_present: int = 3,
                 look_frames: int = 1,
                 z_thr_local: float = None,
                 flux_thr: float = None,
                 confirm_frames: int = None,
                 presence_tpl_fwhm: float = None,
                 presence_halo_fwhm: float = None,
                 presence_halo_fraction: float = None,
                 hold_when_disabled: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if z_thr_local is None or flux_thr is None:
            raise ValueError('z_thr_local and flux_thr are needed (presence and confirmation of a move)')
        if presence and z_thr is None:
            raise ValueError('presence detection needs z_thr (noise-only quantile of the correlation z-score)')
        if int(cmd_latency) < 1:
            raise ValueError('cmd_latency must be >= 1 frame')
        self.n = int(np_sub)
        self.sigma_w = 2.0 * weighted_pix_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))
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
        self.hold_when_disabled = bool(hold_when_disabled)
        self._processed = False
        self.z_thr_local = z_thr_local
        self.flux_thr = flux_thr
        self.confirm_frames = self.block_frames if confirm_frames is None else int(confirm_frames)
        self.cmd_latency = int(cmd_latency)

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
        # presence template (detection): by default the same; set it to the measured PSF for broad PSFs, while the
        # sharp template above keeps the looks and guard C localised (a halo-dominated template smooths the map)
        p_core = tpl_fwhm if presence_tpl_fwhm is None else presence_tpl_fwhm
        p_halo = halo_fwhm if presence_halo_fwhm is None else presence_halo_fwhm
        p_frac = halo_fraction if presence_halo_fraction is None else presence_halo_fraction
        self.separate_presence_tpl = (p_core, p_halo, p_frac) != (tpl_fwhm, halo_fwhm, halo_fraction)
        if self.separate_presence_tpl:
            ptpl = (1.0 - p_frac) * gauss(p_core) + p_frac * gauss(p_halo)
            self.pres_tpl_conj = self.xp.asarray(np.conj(np.fft.fft2(ptpl)).astype(self.complex_dtype))
        else:
            self.pres_tpl_conj = self.tpl_conj
        self.c = (self.n - 1) / 2.0
        self.coord = idx - self.c                                  # pixel index -> px from the centre
        fr = np.fft.fftfreq(self.n)
        self.fx, self.fy = fr[None, :], fr[:, None]

        self.n_moves = 0
        self.n_aborts = 0
        self._mask_w = None
        self.enabled = True
        self.last_look = self.xp.zeros((self.n, self.n), dtype=self.dtype)
        self.reset_state()

        self.inputs['in_pixels'] = InputValue(type=Pixels)
        self.inputs['in_command'] = InputValue(type=BaseValue)
        self.inputs['in_enable'] = InputValue(type=BaseValue, optional=True)
        self.window = BaseValue(value=self.xp.zeros(3, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self.feedforward = BaseValue(value=self.xp.zeros(2, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self.state_out = BaseValue(value=self.xp.zeros(9, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self.outputs['out_window'] = self.window
        self.outputs['out_feedforward'] = self.feedforward
        self.outputs['out_state'] = self.state_out
        self.look_out = BaseValue(value=self.xp.zeros((self.n, self.n), dtype=self.dtype),
                                  target_device_idx=self.target_device_idx)
        self.outputs['out_look'] = self.look_out

    def reset_state(self):
        self.w = np.zeros(2)                       # window centre, px
        self.u_ff = np.zeros(2)                    # accumulated feedforward, px (spot shift)
        self.hist, self.ratios = [], []
        self.hold_until = None
        self.dropout = False
        self.hit = self.miss = 0
        self.acc = None
        self.acc_count = 0
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
        L = self.cmd_latency
        self.cmd_hist = deque(maxlen=L + 1)       # loop command (px) read at steps j-L .. j
        self.uff_hist = deque(maxlen=L)           # feedforward total at the end of steps j-L .. j-1
        self.w_queue = []                         # [(step, window)] returns waiting for the feedforward

    @classmethod
    def input_names(cls):
        return {'in_pixels': InputDesc(Pixels, 'Detector pixels'),
                'in_command': InputDesc(BaseValue, 'Tip/tilt command [nm] as issued by the controller'),
                'in_enable': InputDesc(BaseValue, 'Optional enable flag (> 0.5 = enabled); while disabled the '
                                                  'supervisor only forms the looks')}

    @classmethod
    def output_names(cls):
        return {'out_window': OutputDesc(BaseValue, 'Window centre [wx, wy] px and hold flag'),
                'out_feedforward': OutputDesc(BaseValue, 'Accumulated feedforward tip/tilt command [nm]'),
                'out_look': OutputDesc(BaseValue, 'Image of the last complete look (registered mean) [e-]'),
                'out_state': OutputDesc(BaseValue, 'Telemetry [est_x, est_y, z, guard_ratio, dropout, n_moves, '
                                                   'z_local, window_flux, n_aborts]')}

    # ---- algorithm (host-side decisions on device correlation maps) ----

    def _correlate(self, spectrum):
        return self.xp.real(self.xp.fft.ifft2(spectrum * self.tpl_conj))

    def _correlate_presence(self, spectrum):
        return self.xp.real(self.xp.fft.ifft2(spectrum * self.pres_tpl_conj))

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
        return z > self.z_thr or z_loc > self.z_thr_local or flux > self.flux_thr

    def _update_presence(self, spectrum, corr, flux):
        if self.acc_count == 0:
            self.acc = self.xp.zeros_like(spectrum)
            self.acc_flux = 0.0
        self.acc += spectrum
        self.acc_flux += flux
        self.acc_count += 1
        if self.acc_count < self.block_frames:
            return None
        if self.block_frames == 1:
            cp = self._correlate_presence(spectrum) if self.separate_presence_tpl else corr
        else:
            cp = self._correlate_presence(self.acc / self.block_frames)
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
        u = self._mirror_command(np.asarray(u_loop_px, dtype=float))
        frame = self.xp.asarray(frame, dtype=self.dtype)
        spectrum = self.xp.fft.fft2(frame)
        corr = self._correlate(spectrum)
        if not self.enabled:                                 # passive: looks only, no decision
            look = self._look(spectrum, corr, u)
            est = self._peak(look[0]) if look is not None else np.full(2, np.nan)
            self.uff_hist.append(self.u_ff.copy())
            self.frame += 1
            return dict(est=est, z=None, ratio=np.nan, dropout=False)
        flux = self._window_flux(frame) if (self.presence or self.conf_active) else 0.0
        if self.conf_active and not self.dropout:            # no evidence is collected while blind
            self._update_confirmation(spectrum, flux)
        z = self._update_presence(spectrum, corr, flux) if self.presence else None
        est = np.full(2, np.nan)
        ratio = np.nan
        look = self._look(spectrum, corr, u) if not self.dropout else None
        if look is not None:
            corr_l, u0 = look
            est = self._peak(corr_l)
            self.hist.append(est + u0)
            ratio = self._guard_ratio(corr_l)
            self.ratios.append(ratio)
            self.hist, self.ratios = self.hist[-self.n_cons:], self.ratios[-self.n_cons:]
            if len(self.hist) == self.n_cons:
                arr = np.array(self.hist)
                med = np.median(arr, axis=0)
                # "far" on the latest look in detector coordinates: comparing the median of the registered
                # looks with the CURRENT command adds the disturbance drift over the consensus window
                far = np.linalg.norm(est - self.w) > self.delta
                agree = np.linalg.norm(arr - med, axis=1).max() <= self.delta
                if agree and far and np.median(self.ratios) < self.q_thr and not self.w_queue:
                    self._start_move(med - u)
        self._schedule_feedforward()
        while self.w_queue and self.w_queue[0][0] <= self.frame:
            self._set_window(self.w_queue.pop(0)[1])
        self.uff_hist.append(self.u_ff.copy())
        self.frame += 1
        return dict(est=est, z=z, ratio=ratio, dropout=self.dropout)

    def _mirror_command(self, u_loop):
        """Spot shift (px) produced by what is on the mirror during this frame: loop command + feedforward."""
        self.cmd_hist.append(u_loop)
        L = self.cmd_latency
        cmd = self.cmd_hist[0] if len(self.cmd_hist) == L + 1 else np.zeros(2)
        ff = self.uff_hist[0] if len(self.uff_hist) == L else np.zeros(2)
        return cmd + ff

    def _set_window(self, w_new):
        """Window return: looks taken with the old window (guard ratio, position) must not enter a consensus."""
        self.w = w_new
        self.hist, self.ratios = [], []
        self.look_count = 0

    def _return_window(self, w_new):
        """Window change caused by a feedforward issued now: applied when the feedforward reaches the mirror
        (a window set at step k acts on frame k+1, the feedforward issued at step k lands on frame k+L)."""
        if self.cmd_latency == 1:
            self._set_window(w_new)
        else:
            self.w_queue.append((self.frame + self.cmd_latency - 1, w_new))

    def _look(self, spectrum, corr, u):
        """Correlation map of one look: the frame itself (look_frames = 1) or the mean of look_frames
        frames registered to the command of the first one; None while the look is incomplete."""
        if self.look_frames == 1:
            self.last_look = self.xp.real(self.xp.fft.ifft2(spectrum)).astype(self.dtype)
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
        mean_spectrum = self.look_acc / self.look_frames
        self.last_look = self.xp.real(self.xp.fft.ifft2(mean_spectrum)).astype(self.dtype)
        return self._correlate(mean_spectrum), self.u_look0

    def _update_confirmation(self, spectrum, flux):
        """Accumulate frames at the new window during the hold; decide once confirm_frames are in."""
        if self.confirmed is not None:
            return
        self.conf_acc = spectrum if self.conf_acc is None else self.conf_acc + spectrum
        self.conf_flux += flux
        self.conf_count += 1
        if self.conf_count < self.confirm_frames:
            return
        corr = self._correlate_presence(self.conf_acc / self.conf_count)
        _, z_loc = self._block_stats(corr)
        f = self.conf_flux / self.conf_count
        self.confirmed = z_loc > self.z_thr_local or f > self.flux_thr

    def _start_move(self, xhat):
        """Consensus accepted; ``xhat``: spot position in the detector frame (px)."""
        self.n_moves += 1
        self.hist, self.ratios = [], []
        if not self.conf_active:                      # a new move during a hold keeps the original fallback
            self.w_prev = self.w.copy()
        self.w, self.hold_until = xhat.copy(), self.frame + self.k_hold
        self.conf_active, self.conf_count, self.conf_acc = True, 0, None   # evidence from the next frame on
        self.conf_flux, self.confirmed = 0.0, None

    def _schedule_feedforward(self):
        if self.hold_until is None or self.frame < self.hold_until or self.dropout:
            return
        if self.conf_active:
            if self.confirmed is None:                # confirmation block not complete yet: keep holding
                return
            self.conf_active = False
            if not self.confirmed:                    # wrong consensus: undo the window move, no feedforward
                self.n_aborts += 1
                self.w, self.hold_until = self.w_prev.copy(), None
                return
        self.u_ff = self.u_ff + self.w
        self._return_window(np.zeros(2))
        self.hold_until = None

    def set_enabled(self, enabled):
        """Enable or disable the decisions; on re-enabling restart from the initial state."""
        enabled = bool(enabled)
        if enabled and not self.enabled:
            self.reset_state()
        self.enabled = enabled

    def window_value(self):
        """[wx, wy, hold]: hold during a dropout, and while disabled if hold_when_disabled."""
        hold = self.dropout or (self.hold_when_disabled and not self.enabled)
        return [self.w[0], self.w[1], float(hold)]

    def trigger_code(self):
        if self.local_inputs['in_enable'] is not None:
            self.set_enabled(float(cpuArray(self.local_inputs['in_enable'].value).ravel()[0]) > 0.5)
        # frames only: a step without new pixels (another input refreshed) is not processed
        self._processed = self.local_inputs['in_pixels'].generation_time >= self.current_time
        if not self._processed:
            return
        frame = self.local_inputs['in_pixels'].pixels
        cmd = np.asarray(cpuArray(self.local_inputs['in_command'].value), dtype=float)
        info = self.process_frame(frame, self.cmd_to_px @ cmd[:2])
        ff_nm = self.px_to_cmd @ self.u_ff
        self.window.value[:] = self.xp.asarray(self.window_value(), dtype=self.dtype)
        self.feedforward.value[:] = self.xp.asarray(ff_nm, dtype=self.dtype)
        z, z_loc, f = self.last_block if self.presence else (np.nan, np.nan, np.nan)
        self.state_out.value[:] = self.xp.asarray([*info['est'], z, info['ratio'], float(self.dropout), self.n_moves,
                                                   z_loc, f, self.n_aborts], dtype=self.dtype)
        self.look_out.value[:] = self.last_look

    def post_trigger(self):
        super().post_trigger()
        if not self._processed:                     # no new frame in this step: outputs not refreshed
            return
        for name in ('out_window', 'out_feedforward', 'out_state', 'out_look'):
            self.outputs[name].generation_time = self.current_time
