import numpy as np

from specula.processing_objects.spot_supervisor import SpotSupervisor


class SpotSupervisorSnr(SpotSupervisor):
    """
    :class:`SpotSupervisor` with the locator of :class:`SimpleSpotSupervisor` (SPECULA PR 918) in place of the
    consensus and guard C. Everything else is inherited unchanged: presence logic, dropout and hold, window move,
    confirmation, feedforward and window return (``cmd_latency`` queue).

    Locator: every ``look_frames`` detector frames (one look, not registered with the command), the mean frame minus
    its median is correlated (FFT) with a unit-peak Gaussian of FWHM ``template_fwhm``, and normalised to a
    signal-to-noise ratio with the pixel noise of the look (1.4826 * MAD) times the template norm:

        snr = (img * G) / (1.4826 * MAD(img) * ||G||)

    The global peak p (integer pixel, px from the frame centre) and its SNR s, and the SNR s_loc within
    ``capture_radius`` of the current window w are computed. The window is moved to p (then confirmed and returned as
    in :class:`SpotSupervisor`) if s > ``snr_thr``, |p - w| > ``capture_radius`` and s_loc < ``snr_thr`` (the window
    does not see the spot), and no window return is in flight. No look is taken during a dropout; the look restarts
    at every window change. The consensus parameters (``n_cons``, ``delta_factor``, ``q_thr``, ``r_loc_sigma``) and
    the look template (``tpl_fwhm``, ``halo_fwhm``, ``halo_fraction``) are not used by the looks.

    ``out_state`` keeps the layout of :class:`SpotSupervisor`; its guard-ratio slot holds the SNR s of the last look.
    """

    # explicit signature: Simul checks the parameters of the YAML file against the named arguments of __init__
    def __init__(self,
                 weighted_pix_rad: float,
                 cmd_latency: int,
                 snr_thr: float = 7.5,
                 template_fwhm: float = 3.0,
                 capture_radius: float = 4.0,
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
        super().__init__(weighted_pix_rad=weighted_pix_rad, cmd_latency=cmd_latency, np_sub=np_sub,
                         cmd_to_px=cmd_to_px, ref_offset=ref_offset, n_cons=n_cons, delta_factor=delta_factor,
                         q_thr=q_thr, r_loc_sigma=r_loc_sigma, k_hold=k_hold, tpl_fwhm=tpl_fwhm, halo_fwhm=halo_fwhm,
                         halo_fraction=halo_fraction, presence=presence, z_thr=z_thr, block_frames=block_frames,
                         k_absent=k_absent, k_present=k_present, look_frames=look_frames, z_thr_local=z_thr_local,
                         flux_thr=flux_thr, confirm_frames=confirm_frames, presence_tpl_fwhm=presence_tpl_fwhm,
                         presence_halo_fwhm=presence_halo_fwhm, presence_halo_fraction=presence_halo_fraction,
                         hold_when_disabled=hold_when_disabled, target_device_idx=target_device_idx,
                         precision=precision)
        self.snr_thr = float(snr_thr)
        self.capture_radius = float(capture_radius)
        # unit-peak Gaussian centred on index 0 (wrapped): the correlation peak index of a spot is its pixel index
        idx = np.arange(self.n)
        wrapped = np.minimum(idx, self.n - idx)
        sigma = template_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        g = np.exp(-(wrapped[:, None] ** 2 + wrapped[None, :] ** 2) / (2.0 * sigma ** 2))
        self.g_norm = float(np.sqrt(np.sum(g ** 2)))
        self.g_conj = self.xp.asarray(np.conj(np.fft.fft2(g)).astype(self.complex_dtype))
        self._loc = None                                          # (window, flat indices of the local disk)

    def _local_idx(self):
        """Flat indices of the pixels within capture_radius of the current window, cached per window."""
        key = (float(self.w[0]), float(self.w[1]))
        if self._loc is None or self._loc[0] != key:
            d2 = (self.coord[None, :] - key[0]) ** 2 + (self.coord[:, None] - key[1]) ** 2
            self._loc = (key, self.xp.asarray(np.flatnonzero(d2 <= self.capture_radius ** 2)))
        return self._loc[1]

    def _snr_look(self, image):
        """(p [px from the centre], s, s_loc) of one look image."""
        xp = self.xp
        med = xp.median(image)
        noise = 1.4826 * xp.median(xp.abs(image - med))
        snr = xp.real(xp.fft.ifft2(xp.fft.fft2(image - med) * self.g_conj)) / (noise * self.g_norm)
        i = int(xp.argmax(snr))
        row, col = divmod(i, self.n)
        s_loc = float(xp.max(snr.ravel()[self._local_idx()]))
        return np.array([col - self.c, row - self.c]), float(snr.ravel()[i]), s_loc

    def _accumulate(self, frame):
        """Mean of look_frames frames (plain sum, not registered); None while the look is incomplete."""
        if self.look_count == 0:
            self.look_acc = self.xp.zeros_like(frame)
        self.look_acc += frame
        self.look_count += 1
        if self.look_count < self.look_frames:
            return None
        self.look_count = 0
        self.last_look = self.look_acc / self.look_frames
        return self.last_look

    def _start_move(self, xhat):
        super()._start_move(xhat)
        self.look_count = 0                                       # the next look starts at the new window

    def process_frame(self, frame, u_loop_px):
        """One detector frame; same bookkeeping as SpotSupervisor.process_frame, with the SNR locator."""
        self._mirror_command(np.asarray(u_loop_px, dtype=float))  # keeps the feedforward history in step
        frame = self.xp.asarray(frame, dtype=self.dtype)
        est, ratio = np.full(2, np.nan), np.nan
        if not self.enabled:                                      # passive: looks only, no decision
            img = self._accumulate(frame)
            if img is not None:
                est, ratio, _ = self._snr_look(img)
            self.uff_hist.append(self.u_ff.copy())
            self.frame += 1
            return dict(est=est, z=None, ratio=ratio, dropout=False)
        spectrum = self.xp.fft.fft2(frame) if (self.presence or self.conf_active) else None
        flux = self._window_flux(frame) if (self.presence or self.conf_active) else 0.0
        if self.conf_active and not self.dropout:
            self._update_confirmation(spectrum, flux)
        z = None
        if self.presence:
            # corr is used by the presence block only with block_frames 1 and no separate presence template
            corr = self._correlate_presence(spectrum) if self.block_frames == 1 else None
            z = self._update_presence(spectrum, corr, flux)
        img = self._accumulate(frame) if not self.dropout else None
        if img is not None:
            est, ratio, s_loc = self._snr_look(img)
            far = np.linalg.norm(est - self.w) > self.capture_radius
            if ratio > self.snr_thr and far and s_loc < self.snr_thr and not self.w_queue:
                self._start_move(est)
        self._schedule_feedforward()
        while self.w_queue and self.w_queue[0][0] <= self.frame:
            self._set_window(self.w_queue.pop(0)[1])
        self.uff_hist.append(self.u_ff.copy())
        self.frame += 1
        return dict(est=est, z=z, ratio=ratio, dropout=self.dropout)
