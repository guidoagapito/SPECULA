import unittest
import os
import shutil
import subprocess
import sys
import glob
import specula
specula.init(-1,precision=1)  # Default target device

from specula import np
from specula.simul import Simul
from astropy.io import fits

class TestPyramidSimulation(unittest.TestCase):
    """Test Pyramid simulation by running a full simulation and checking the results"""

    def setUp(self):
        """Set up test by ensuring calibration directory exists"""
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')

        # Get current working directory
        self.cwd = os.getcwd()

    def tearDown(self):
        """Clean up after test by removing generated files"""
        # Remove test/data directory with timestamp
        data_dirs = glob.glob(os.path.join(self.datadir, '2*'))
        for data_dir in data_dirs:
            if os.path.isdir(data_dir) and os.path.exists(f"{data_dir}/intensity.fits"):
                shutil.rmtree(data_dir)

        # Change back to original directory
        os.chdir(self.cwd)

    def test_pyramid_simulation(self):
        """Run the simulation and check the results"""

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Run the simulation
        print("Running Pyramid simulation...")
        yml_files = ['params_pyr_ol_test.yml']
        simul = Simul(*yml_files)
        simul.run()

        # Find the most recent data directory (with timestamp)
        data_dirs = sorted(glob.glob(os.path.join(self.datadir, '2*')))
        print(f"Data directories found: {data_dirs}")
        self.assertTrue(data_dirs, "No data directory found after simulation")
        latest_data_dir = data_dirs[-1]

        # Check if intensity.fits exists
        intensity_path = os.path.join(latest_data_dir, 'intensity.fits')
        self.assertTrue(os.path.exists(intensity_path),
                       f"intensity.fits not found in {latest_data_dir}")
        ccd_path = os.path.join(latest_data_dir, 'ccd.fits')
        self.assertTrue(os.path.exists(ccd_path),
                       f"ccd.fits not found in {latest_data_dir}")

        # Verify intensity values are within expected range
        # and the size is correct [10,60,60]
        with fits.open(intensity_path) as hdul:
            # Check if there's data
            self.assertTrue(len(hdul) > 0, "No data found in intensity.fits")
            data = hdul[0].data
            self.assertIsNotNone(data, "Data in intensity.fits is None")
            self.assertTrue(np.all(data >= 0), "Intensity values are negative")
            self.assertEqual(data.shape, (10, 60, 60), "Intensity data shape is incorrect")