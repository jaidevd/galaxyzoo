# stdlib imports
import unittest
import json
import os
import tempfile
import shutil

# system library imports
import matplotlib.pyplot as plt

# Local imports
from galaxyzoo.ui.data_inspector import DataInspector
from galaxyzoo.processing.api import get_largest_region
from galaxyzoo.processing.transform_images import process_image


ROOT = "/Users/jaidevd/GitHub/kaggle/galaxyzoo"

class TestUI(unittest.TestCase):
    """
    Tests for the data exploration app.
    """
    @classmethod
    def setUpClass(cls):
        cls.di = DataInspector()
        cls.defects = os.path.join(ROOT, 'defective_files.json')
        cls.processed_path = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.processed_path)

    def test_dataframe_adapter(self):
        adapter = self.di.adapter
        self.assertEqual(len(adapter.columns), 37)
        row_label = adapter.get_row_label(0)
        ind = self.di.solutions.index[0]
        self.assertEqual(str(ind), row_label)
        text = adapter.get_text(self.di,'solutions', 0, 0)
        self.assertAlmostEqual(float(text), adapter.data.irow(0)[0], 3)
        cols = [col[5:] for col in self.di.solutions.columns]
        self.assertItemsEqual(cols, adapter.columns)

    def test_image_container(self):
        self.assertEqual(len(self.di.image.plot_components), 3)
        bottom, middle, top = self.di.image.plot_components
        for value in bottom.data.arrays.itervalues():
            self.assertEqual(len(value), 37)
        im_org = middle.data.arrays.get('im')
        self.assertTupleEqual(im_org.shape, (424,424,3))
        im_proc = top.data.arrays.get('im')
        self.assertTupleEqual(im_proc.shape, (128,128))

    def test_processed_image(self):
        largest = None
        while largest is None:
            self.di.data_id = self.di.get_random_image()
            top = self.di.image.plot_components[2]
            im = top.data.arrays.get('im')
            # Assert that the orientation is indeed zero
            largest = get_largest_region(im)
        self.assertLessEqual(round(largest.orientation), 1)

    def test_defective_images(self):
        """
        Assert that the images in the defective list are indeed defective.
        """
        with open(self.defects, 'r') as f:
            defects_list = json.load(f)
            defects = [f[0] for f in defects_list]
        isError = []
        for defect in defects:
            path = os.path.join(ROOT, 'images_training',defect)
            try:
                x = plt.imread(path)
                cropped = process_image(x)
                save_path = os.path.join(self.processed_path, defect)
                plt.imsave(save_path, cropped, cmap=plt.cm.gray)
            except:
                isError.append(True)
        self.assertTrue(all(isError))

    def test_get_data_by_id(self):
        data_id = self.di.get_random_image()
        pdf, im = self.di.get_data_by_id(data_id)
        self.assertTupleEqual(im.shape, (424, 424, 3))
        self.assertTupleEqual(pdf.shape, (37,))

    def test_update_cropped_image_data(self):
        org_image = self.di.image_data.arrays.get('im')[:,:,0]
        self.di.data_id = self.di.get_random_image()
        new_image = self.di.image_data.arrays.get('im')[:,:,0]
        diff = org_image == new_image
        total = diff.sum().sum()
        self.assertLess(total, (424**2)/10.0)

    def test_update_labeled_image_data(self):
        org_apd = self.di.labeled_image_plot.data.arrays
        self.di.data_id = self.di.get_random_image()
        new_apd = self.di.labeled_image_plot.data.arrays
        self.assertNotEqual(len(org_apd), len(new_apd))


if __name__ == "__main__":
    unittest.main()