import unittest
import os
import shutil
import specula
specula.init(-1, precision=1)

from specula.simul import Simul
from astropy.io import fits
import numpy as np

class TestPyrPupdataCalibration(unittest.TestCase):
    """Test Pyramid PupData calibration by comparing generated calibration files with reference ones"""

    def setUp(self):
        self.testdir = os.path.dirname(__file__)
        self.calibdir = os.path.join(self.testdir, 'calib')
        self.datadir = os.path.join(self.testdir, 'data')
        os.makedirs(self.calibdir, exist_ok=True)
        os.makedirs(self.datadir, exist_ok=True)

        self.pupdata_ref_path = os.path.join(self.datadir, 'scao_pupdata_ref.fits')
        self.pupdata_path = os.path.join(self.calibdir, 'scao_pupdata.fits')

        self._cleanFiles()
        self.cwd = os.getcwd()

    def _cleanFiles(self):
        if os.path.exists(self.pupdata_path):
            os.remove(self.pupdata_path)

    def tearDown(self):
        self._cleanFiles()
        os.chdir(self.cwd)

    def test_pyr_pupdata_calibration(self):
        """Test Pyramid PupData calibration by comparing generated calibration file with reference"""

        os.chdir(self.testdir)

        # Check if reference file exists
        self.assertTrue(os.path.exists(self.pupdata_ref_path), f"Reference file {self.pupdata_ref_path} does not exist")

        # Run the simulation for calibration
        yml_files = [
            os.path.join(self.testdir, '../main/scao/params_scao.yml'),
            os.path.join(self.testdir, '../main/scao/calib_pyr_pupdata.yml')
        ]
        simul = Simul(*yml_files)
        simul.run()

        # Check if the calibration file was generated
        self.assertTrue(os.path.exists(self.pupdata_path), "Pyramid PupData calibration file was not generated")

        # Compare the generated file with the reference file
        with fits.open(self.pupdata_path) as gen_pup:
            with fits.open(self.pupdata_ref_path) as ref_pup:
                for i, (gen_hdu, ref_hdu) in enumerate(zip(gen_pup, ref_pup)):
                    if hasattr(gen_hdu, 'data') and hasattr(ref_hdu, 'data') and gen_hdu.data is not None:
                        np.testing.assert_array_almost_equal(
                            gen_hdu.data, ref_hdu.data,
                            decimal=5,
                            err_msg=f"Data in HDU #{i} does not match reference"
                        )

        print("Pyramid PupData calibration matches reference!")

    @unittest.skip("This test is only used to create reference files")
    def test_create_reference_file(self):
        """Create reference file for Pyramid PupData calibration"""

        os.chdir(self.testdir)

        # Run the simulation for calibration
        yml_files = [
            os.path.join(self.testdir, '../main/scao/params_scao.yml'),
            os.path.join(self.testdir, '../main/scao/calib_pyr_pupdata.yml')
        ]
        simul = Simul(*yml_files)
        simul.run()

        # Check if the calibration file was generated
        self.assertTrue(os.path.exists(self.pupdata_path), "Pyramid PupData calibration file was not generated")

        # Copy file to reference directory
        shutil.copy(self.pupdata_path, self.pupdata_ref_path)
        print("Reference file created and saved to test/data/")
        print("Please commit this file to the repository for future tests")