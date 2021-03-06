# stdlib imports
import unittest
import os
import tempfile
import shutil

# system library imports
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import circle, ellipse
from nose import SkipTest

# Local imports
import galaxyzoo.processing.api as gzapi
from galaxyzoo.processing.transform_images import process_image


def pipeline(filename, temp_dir):
    """
    Convenient wrapper around the feature extraction pipeline
    """
    x = plt.imread(filename)[:,:,0]
    cropped = process_image(x)
    to_save = filename.split('.')[0] + '.png'
    plt.imsave(os.path.join(temp_dir, to_save), cropped,
               cmap=plt.cm.gray)


class TestAPI(unittest.TestCase):
    """
    Tests for the api module.
    """

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        ind = np.random.randint(0,len(gzapi.all_files))
        self.test_id = int(gzapi.all_files[ind].split('.')[0])

    def test_get_data_by_id(self):
        """
        Test whether the `get_data_by_id` function works well.
        """
        im, pdf = gzapi.get_data_by_id(self.test_id, False)
        self.assertEqual(im.ndim, 3)
        im = gzapi.get_data_by_id(self.test_id, True)[0]
        self.assertEqual(im.ndim, 2)
        expected_pdf = gzapi.solutions.ix[self.test_id].values
        expected_im = plt.imread(os.path.join(gzapi.TRAINING_IMAGES_DIR,
                                              str(self.test_id)+'.jpg'))[:,:,0]
        self.assertTrue(np.allclose(expected_im, im))
        self.assertTrue(np.allclose(expected_pdf, pdf))

    def test_defective_indices(self):
        """
        Test whether the defective files are indeed defective
        """
        errors = (IOError)
        defects = gzapi.get_defective_files()
        for filename in defects:
            full_path = os.path.join(gzapi.TRAINING_IMAGES_DIR, filename[0])
            self.assertTrue(os.path.isfile(full_path))

    def test_get_processed_image(self):
        """
        Test whether the get_processed_image function works
        """
        im = gzapi.get_processed_image(self.test_id, as_gray=False)
        self.assertEqual(im.ndim, 3)
        im = gzapi.get_processed_image(self.test_id, as_gray=True)
        self.assertEqual(im.ndim, 2)
        self.assertTupleEqual(im.shape, (128, 128))

    def test_thresholding(self):
        """
        test whether the thresholding works
        :return:
        """
        x = gzapi.get_data_by_id(self.test_id, as_grey=True)[0]
        bw = gzapi.get_thresholded_image(x)
        unique_levels = np.unique(bw)
        self.assertEqual(len(unique_levels), 2)

    def test_clear_borders(self):
        x = gzapi.get_data_by_id(self.test_id, as_grey=True)[0]
        bw = gzapi.get_cleared_binary_image(x)
        top = bw[0,:]
        bottom = bw[bw.shape[0]-1,:]
        left = bw[:,0]
        right = bw[:,bw.shape[1]-1]
        for col in (top, bottom, left, right):
            self.assertTrue(np.allclose(col, np.zeros(col.shape)))

    def test_get_region_bounds(self):
        test_img = np.zeros((1000,1000), dtype=np.uint8)
        rr, cc = circle(100,100,50)
        test_img[rr,cc] = 1
        bx, by = gzapi.get_region_bounds(test_img)
        bx, by = bx[0], by[0]
        minc, maxc = bx
        minr, maxr = by
        self.assertEqual(maxc - minc + 1, 100)
        self.assertEqual(maxr - minr + 1, 100)
        center = gzapi.get_bbox_center((minr, minc, maxr, maxc))
        self.assertTupleEqual(center, (100,100))

    def test_get_largest_region(self):
        test_img = np.zeros((1000,1000), dtype=np.uint8)
        rr, cc = circle(100,100,50)
        test_img[rr,cc] = 1
        rr, cc = circle(500,500,100)
        test_img[rr, cc] = 1
        largest_region = gzapi.get_largest_region(test_img)
        self.assertTupleEqual(largest_region.centroid, (500,500))

    def test_rotate_largest_region(self):
        test_img = np.zeros((1000,1000), dtype=np.uint8)
        rr, cc = circle(100,100,50)
        test_img[rr,cc] = 1
        rr, cc = ellipse(500,500,300,100)
        test_img[rr, cc] = 1
        rotated = gzapi.rotate_largest_region(test_img)
        largest_region = gzapi.get_largest_region(rotated)
        self.assertAlmostEqual(largest_region.orientation, 0)
        self.assertIsNone(gzapi.rotate_largest_region(None))

    def test_point_rotate(self):
        x, y = 1.0, 1.0
        theta = np.pi/2
        x_rot, y_rot = gzapi.point_rotate(x, y, theta)
        self.assertAlmostEqual(x_rot, -1)
        self.assertAlmostEqual(y_rot, 1)

    def test_create_matrix_from_images(self):
        raise SkipTest()
        # test the full blown function, read all images and all indices
        indices, images = gzapi.create_matrix_from_images()
        self.assertEqual(len(indices), images.shape[0])

    def test_create_matrix_from_images_random(self):
        # Create a matrix out of randomly selected indices
        indices, images = gzapi.create_matrix_from_images(random=True,
                                                          n_images=100)
        self.assertEqual(len(indices), 100)
        self.assertEqual(images.shape[0], 100)
        sample_inds = np.random.randint(0,len(indices),(3,))
        index_samples = indices[sample_inds]
        image_samples = images[sample_inds,:]
        for i in range(3):
            impath = os.path.join(gzapi.PROCESSED_IMAGES_DIR, 
                                  str(index_samples[i])+'.jpg')
            x = plt.imread(impath)[:,:,0].ravel()
            y = image_samples[i,:]
            self.assertTrue(np.allclose(x,y))
        self.assertRaises(ValueError, gzapi.create_matrix_from_images, True)

    def test_create_matrix_from_specific_indices(self):
        # create a matrix out of specific indices
        df = gzapi.solutions.values[:,0:3]
        inds = np.argmax(df, axis=1)
        smooth = inds == 0
        smooth_image_indices = gzapi.solutions.index[smooth].values
        image_indices, X =gzapi.create_matrix_from_images(
                                                indices=smooth_image_indices)
        self.assertEqual(len(smooth_image_indices), len(image_indices))
        self.assertEqual(len(smooth_image_indices), X.shape[0])
        self.assertTrue(np.allclose(smooth_image_indices, image_indices))

    def test_solution_data_first_three_columns(self):
        df = gzapi.solutions
        cols = df.columns[:3]
        s = np.zeros((df.shape[0]))
        for col in cols:
            s += df[col].values
        self.assertTrue(np.allclose(s, np.ones((s.shape[0],))))

    def test_solution_data_class7(self):
        df = gzapi.solutions
        x = df['Class1.1'].values
        y = df['Class7.1'] + df['Class7.2'] + df['Class7.3']
        y = y.values
        self.assertTrue(np.allclose(x, y, 1e-4))

if __name__ == "__main__":
    unittest.main()
