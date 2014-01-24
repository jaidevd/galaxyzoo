# Standard library imports
import os
import json

# System libray imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import closing, square, label
from skimage.segmentation import clear_border
from skimage.filter import threshold_otsu
from skimage.measure import regionprops
from skimage.transform import rotate
import tables


TRAINING_IMAGES_DIR = "/Users/jaidevd/GitHub/kaggle/galaxyzoo/images_training_rev1"
TRAINING_SOLN_PATH = "/Users/jaidevd/GitHub/kaggle/galaxyzoo/training_solutions_rev1.csv"
PROCESSED_IMAGES_DIR = "/Users/jaidevd/GitHub/kaggle/galaxyzoo/processed_images"
DEFECTS = '/Users/jaidevd/GitHub/kaggle/galaxyzoo/defective_files.json'

solutions = pd.read_csv(TRAINING_SOLN_PATH, index_col=0)
all_files = [f for f in os.listdir((PROCESSED_IMAGES_DIR)) if f.endswith('jpg')]


def get_data_by_id(galaxy_id, as_grey=False, dir=TRAINING_IMAGES_DIR):
    """
    Get the training data by the galaxy id

    :param galaxy_id: ID of the galaxy
    :as_grey: If true, the image is returned as a greyscale image

    :returns:
    --------
    im : np.ndarray
        The image array of the galaxy

    pdf : np.ndarray
        The probably distribution of the galaxy classifications.

    """
    impath = os.path.join(dir, str(galaxy_id) + '.jpg')
    im = plt.imread(impath)
    if as_grey:
        im = im[:, :, 0]
    pdf = solutions.ix[galaxy_id].values
    return im, pdf


def get_defective_files():
    """
    :returns: list of IDs that are defective.
    """
    with open(DEFECTS,'r') as f:
        defects = json.load(f)
    return defects


def get_processed_image(galaxy_id, as_gray=False):
    """
    Return the processed image by ID

    :param galaxy_id: ID of the image

    :param as_gray: If true, the image is returned as a grayscale image.

    :returns:
    -----------
    im : ndarray
        The image array
    """
    impath = os.path.join(PROCESSED_IMAGES_DIR,str(galaxy_id) + '.jpg')
    im = plt.imread(impath)
    if as_gray:
        im = im[:,:,0]
    return im


def get_thresholded_image(x):
    """
    For a given image array, calculates the threshold and returns the binary
    thresholded image

    :param x: ndarray

    :returns:
    ---------
    ndarray, boolean
    """
    if x is not None:
        thresh = threshold_otsu(x)
        bw = closing(x > thresh, square(1))
        return bw
    return None


def get_cleared_binary_image(x):
    """
    Given an image array, return the binary thresholded image and clear the
    border regions.
    :param x: ndarray
    :return:
    bw: ndarray of type bool
    """
    if x is not None:
        bw = get_thresholded_image(x)
        bw = clear_border(bw)
        return bw
    return None


def get_region_bounds(x):
    """ Given an image array, return the bounding boxes of all regions in it.
    Input image MUST be a binary image."""
    bw = label(x)
    regions = regionprops(bw)
    bx = []
    by = []
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        bx.append((minc, maxc))
        by.append((minr, maxr))
    return bx, by


def get_bbox_center(bbox):
    """
    Get the center of the rectangle formed by the tuple bbox
    :param bbox: tuple of the form (min_row, min_col, max_row, max_col)
    :return: tuple of the form (row_center, col_center)
    """
    minr, minc, maxr, maxc = bbox
    rr = (minr + maxr) / 2
    cc = (minc + maxc) / 2
    return (rr, cc)


def get_largest_region(x):
    """
    For the image x, get all morphological regions and return the one with
    the largest area.
    :param x: ndarray
    :return: skimage.measure._RegionProperties
    """
    if x is not None:
        bw = get_cleared_binary_image(x)
        labeled = label(bw)
        regions = regionprops(labeled)
        bounds = [prop.bbox for prop in regions]
        areas = np.array([abs(t[0] - t[2])*abs(t[1] - t[3]) for t in bounds])
        if len(areas)!=0:
            largest_region = regions[np.argmax(areas)]
            return largest_region
    return None


def rotate_largest_region(x):
    """
    Get the largest mophological region in image x and rotate the image
    counter-clockwise through an angle equal to the orientation of the
    largest region.
    :param x: ndarray
    :return: ndarray
    """
    largest_region = get_largest_region(x)
    if largest_region is not None:
        orientation = largest_region.orientation
        rotated_image = rotate(x, angle=-np.rad2deg(orientation), resize=True)
        return rotated_image
    return None


def point_rotate(x, y, theta):
    """
    Rotate the point (x,y) counter-clockwise through the angle theta on the
    Cartesian plane.
    :param x: Float
    :param y: Float
    :param theta: Float, angle in radians
    :return: array([x_rot, y_rot]), coordinates of the rotated point.
    """
    rotmat = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
    return np.dot(rotmat, np.array([[x],[y]])).ravel()


def crop_around_centroid(image, rr, cc, rows=128, cols=128):
    """
    Crop the image such that the center is at [rr, cc] and the shape of the
    image is (rows, cols)
    :param image: ndarray
    :param rr: central row of the cropped image
    :param cc: central column of the cropped image
    :param rows: number of rows to crop
    :param cols: number of columns to crop
    :return: ndarray
    """
    rmin = rr - rows/2
    rmax = rr + rows/2
    cmin = cc - cols/2
    cmax = cc + cols/2
    cropped = image[rmin:rmax, cmin:cmax]
    return cropped

def create_matrix_from_images(random=False, indices=None, n_images=None):
    """
    Create a matrix from the processed images, such that each row of the matrix
    is a flattened image array.
    :param random: If True, `n_images` are selected at random from the dataset
    and `indices` is ignored.
    :param indices: If not None, this should be a list or array of integer
    indices that specify which images to choose for the array.
    :return X: ndarray
    """
    all_images = os.listdir(PROCESSED_IMAGES_DIR)
    defects = get_defective_files()
    for image in defects:
        if image[0] in all_images:
            all_images.remove(image[0])
    all_images = np.array(all_images)
    image_indices = []
    if random:
        if n_images is not None:
            X = np.zeros((n_images, 128**2))
            inds = np.random.randint(0, len(all_images), (n_images,))
            samples = all_images[inds]
            for i in range(X.shape[0]):
                image_indices.append(samples[i].split('.')[0])
                impath = os.path.join(PROCESSED_IMAGES_DIR, samples[i])
                try:
                    x = plt.imread(impath)[:,:,0]
                except Exception, err:
                    x = np.ones((128**2,))
                    defects.append((samples[i], str(err)))
                X[i,:len(x.ravel())] = x.ravel()
        else:
            raise ValueError("n_images cannot be None.")
    else:
        if indices is not None:
            indices = map(str, indices)
            indices = [ind+'.jpg' for ind in indices]
            X = np.zeros((len(indices),128**2))
            for i in range(X.shape[0]):
                image_indices.append(indices[i].split('.')[0])
                impath = os.path.join(PROCESSED_IMAGES_DIR, indices[i])
                try:
                    x = plt.imread(impath)[:,:,0]
                except Exception, err:
                    x = np.ones((128**2,))
                    defects.append((indices[i], str(err)))
                X[i,:len(x.ravel())] = x.ravel()
        else:
            X = np.zeros((len(all_images), 128**2))
            for i in range(X.shape[0]):
                image_indices.append(all_images[i].split('.')[0])
                impath = os.path.join(PROCESSED_IMAGES_DIR,all_images[i])
                try:
                    x = plt.imread(impath)[:,:,0]
                except Exception, err:
                    x = np.ones((128**2,))
                    defects.append((samples[i], str(err)))
                X[i,:len(x.ravel())] = x.ravel()
    with open(DEFECTS, 'w') as f:
        json.dump(defects, f)
    assert len(image_indices) == X.shape[0]
    image_indices = map(int, image_indices)
    image_indices = np.array(image_indices, dtype=np.int32)
    return image_indices, X
            

def store_hd5(name, data):
    """
    A convenience function to store numpy arrays as HDF5 file
    :param name: Name of the file to write to.
    :param kwargs: names and values of numpy arrays to be stored.
    """
    f = tables.openFile(name)
    for key, value in data.iteritems():
        atom = tables.Atom.from_dtype(value.dtype)
        x = f.createCArray(f.root, key, atom, value.shape)
        x[:] = value
    f.close()

def get_hdf(name, keys):
    f = tables.openFile(name)
    arrays = []
    for key in keys:
        node = f.getNode(key)
        data = node.read()
        arrays.append(data)
    f.close()
    return arrays
