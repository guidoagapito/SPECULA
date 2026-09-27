import specula
specula.init(0)  # Default target device

import unittest

from specula import np

from specula.processing_objects.spot_supervisor import SpotSupervisor

N = 64
RADIUS = 2.0


def broad_spot(amp=40.0, core=4.0, halo=14.0, frac=0.8):
    c = (N - 1) / 2.0
    r2 = (np.arange(N)[:, None] - c) ** 2 + (np.arange(N)[None, :] - c) ** 2
    k = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    g = lambda fwhm: np.exp(-r2 / (2.0 * (fwhm * k) ** 2)) / fwhm ** 2
    return amp * ((1 - frac) * g(core) + frac * g(halo)) / ((1 - frac) * g(core) + frac * g(halo)).max()


def make(**kw):
    return SpotSupervisor(weighted_pix_rad=RADIUS, np_sub=N, target_device_idx=-1, presence=True, z_thr=0.0,
                          block_frames=4, **kw)


class TestPresenceTemplate(unittest.TestCase):

    def test_default_is_the_main_template(self):
        sup = make()
        self.assertFalse(sup.separate_presence_tpl)
        self.assertIs(sup.pres_tpl_conj, sup.tpl_conj)

    def test_same_values_are_not_separate(self):
        sup = make(presence_tpl_fwhm=2.0, presence_halo_fwhm=15.0, presence_halo_fraction=0.3)
        self.assertFalse(sup.separate_presence_tpl)

    def test_matched_presence_template_raises_z_for_a_faint_broad_psf(self):
        """Faint broad star (the regime where presence matters): the PSF-matched template gives a higher z on
        average. For bright stars the self-normalised z of a broad template saturates (the star's own halo inflates
        the map std) -- irrelevant there, the star is present by a wide margin anyway."""
        mean_z = {}
        for name, kw in (("fixed", {}), ("matched", dict(presence_tpl_fwhm=4.0, presence_halo_fwhm=14.0,
                                                           presence_halo_fraction=0.8))):
            zs = []
            for seed in range(10):
                rng = np.random.default_rng(seed)
                sup = make(**kw)
                for _ in range(4):
                    sup.process_frame(broad_spot(amp=0.5) + rng.standard_normal((N, N)), np.zeros(2))
                zs.append(sup.last_block[0])
            mean_z[name] = np.mean(zs)
        self.assertGreater(mean_z["matched"], mean_z["fixed"])

    def test_looks_keep_the_sharp_template(self):
        a = make()
        b = make(presence_tpl_fwhm=4.0, presence_halo_fwhm=14.0, presence_halo_fraction=0.8)
        np.testing.assert_array_equal(a.tpl_conj, b.tpl_conj)
        self.assertFalse(np.array_equal(a.tpl_conj, b.pres_tpl_conj))

    def test_single_frame_blocks_use_the_presence_template(self):
        rng = np.random.default_rng(4)
        f = broad_spot() + rng.standard_normal((N, N))
        kw = dict(presence_tpl_fwhm=4.0, presence_halo_fwhm=14.0, presence_halo_fraction=0.8)
        s1 = SpotSupervisor(weighted_pix_rad=RADIUS, np_sub=N, target_device_idx=-1, presence=True, z_thr=0.0,
                            block_frames=1, **kw)
        s4 = SpotSupervisor(weighted_pix_rad=RADIUS, np_sub=N, target_device_idx=-1, presence=True, z_thr=0.0,
                            block_frames=4, **kw)
        s1.process_frame(f, np.zeros(2))
        for _ in range(4):
            s4.process_frame(f, np.zeros(2))                       # same frame four times: same block mean
        self.assertAlmostEqual(s1.last_block[0], s4.last_block[0], places=3)


if __name__ == '__main__':
    unittest.main()
