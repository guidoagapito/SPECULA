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

    ``ref_offset`` is the offset of the matched-filter peak from the WCoG centroid for a spot at the
    reference (peak/centroid asymmetry, integer-pixel quantisation): the estimates are the centroid
    displacement from the reference, so the window centre ``w`` needs no further correction.

    Coordinates: x = column, y = row, in pixels from the subaperture centre. ``in_command`` is the
    tip/tilt command really applied during the frame (nm); ``cmd_to_px`` maps it to the spot shift
    produced by the correction (px per nm, sign and axes to be calibrated on the system).

    Outputs: ``out_window`` = [wx, wy, hold] (feed it to ShSlopecMovable, delayed by one step),
    ``out_feedforward`` = accumulated feedforward command (nm, feed it to an extra tip/tilt DM),
    ``out_state`` = telemetry [est_x, est_y, z, guard_ratio, dropout, n_moves].
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
        self.reset_state()

        self.inputs['in_pixels'] = InputValue(type=Pixels)
        self.inputs['in_command'] = InputValue(type=BaseValue)
        self.window = BaseValue(value=self.xp.zeros(3, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self.feedforward = BaseValue(value=self.xp.zeros(2, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self.state_out = BaseValue(value=self.xp.zeros(6, dtype=self.dtype), target_device_idx=self.target_device_idx)
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
        self.frame = 0

    @classmethod
    def input_names(cls):
        return {'in_pixels': InputDesc(Pixels, 'Detector pixels'),
                'in_command': InputDesc(BaseValue, 'Tip/tilt command applied during the frame [nm]')}

    @classmethod
    def output_names(cls):
        return {'out_window': OutputDesc(BaseValue, 'Window centre [wx, wy] px and hold flag'),
                'out_feedforward': OutputDesc(BaseValue, 'Accumulated feedforward tip/tilt command [nm]'),
                'out_state': OutputDesc(BaseValue, 'Telemetry [est_x, est_y, z, guard_ratio, dropout, n_moves]')}

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

    def _update_presence(self, spectrum, corr, u):
        if self.acc_count == 0:
            self.u_block0 = u.copy()
            self.acc = self.xp.zeros_like(spectrum)
        du = u - self.u_block0
        phase = np.exp(-2j * np.pi * (self.fx * du[0] + self.fy * du[1]))
        self.acc += spectrum * self.xp.asarray(phase.astype(self.complex_dtype))
        self.acc_count += 1
        if self.acc_count < self.block_frames:
            return None
        cp = corr if self.block_frames == 1 else self._correlate(self.acc / self.block_frames)
        z = float((cp.max() - cp.mean()) / cp.std())
        self.acc_count = 0
        if z > self.z_thr:
            self.hit, self.miss = self.hit + 1, 0
        else:
            self.hit, self.miss = 0, self.miss + 1
        if not self.dropout and self.miss >= self.k_absent:
            self.dropout = True
            self.look_count = 0
        elif self.dropout and self.hit >= self.k_present:
            self.dropout = False
            self.hist, self.ratios = [], []
            self.look_count = 0
        return z

    def process_frame(self, frame, u_loop_px):
        """One detector frame. ``u_loop_px``: spot shift (px) produced by the loop command applied
        during the frame; the supervisor adds its own feedforward. Returns a telemetry dict."""
        u = np.asarray(u_loop_px, dtype=float) + self.u_ff
        spectrum = self.xp.fft.fft2(self.xp.asarray(frame, dtype=self.dtype))
        corr = self._correlate(spectrum)
        z = self._update_presence(spectrum, corr, u) if self.presence else None
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

    def _start_move(self, xhat):
        """Consensus accepted; ``xhat``: spot position in the detector frame (px)."""
        self.n_moves += 1
        self.hist, self.ratios = [], []
        self.n3_left = 0
        if self.ff_mode == 'one':
            self.ff_pending, self.w = xhat.copy(), np.zeros(2)
        else:
            self.w, self.hold_until = xhat.copy(), self.frame + self.k_hold

    def _schedule_feedforward(self):
        if self.ff_pending is not None:
            self.u_ff = self.u_ff + self.ff_pending
            self.ff_pending = None
        elif self.hold_until is not None and self.frame >= self.hold_until and not self.dropout:
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
        z = np.nan if info['z'] is None else info['z']
        self.state_out.value[:] = self.xp.asarray([*info['est'], z, info['ratio'], float(self.dropout), self.n_moves],
                                                  dtype=self.dtype)

    def post_trigger(self):
        super().post_trigger()
        for name in ('out_window', 'out_feedforward', 'out_state'):
            self.outputs[name].generation_time = self.current_time
