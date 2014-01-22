################################################################################
# Description:
# -------------
# This is a TraistUI (http://docs.enthought.com/traitsui) application that 
# is used for performing some exploratory data analysis on the training data
# from Kaggle's Galaxy Zoo competition. 
# (http://kaggle.com/c/galaxy-zoo-the-galaxy-challenge)
# Broadly, it can be thought of as a browser for the data, which shows the 
# training images, the target distribution functions and some metrics related 
# to the training data.
# ------------------------------------------------------------------------------
# Author: Jaidev Deshpande
# email: deshpande.jaidev@gmail.com
# ------------------------------------------------------------------------------
# Dependencies: 
# -------------
# 1. NumPy
# 2. SciPy
# 3. matplotlib
# 4. Traits
# 5. TraitsUI
# 6. Chaco
# 7. scikit-image
# 8. scikit-learn
# 9. pandas
################################################################################

# stdlib imports
import os

# system library imports
import numpy as np
import pandas as pd
from skimage.exposure import histogram
from skimage.io import imread

# ETS imports
from traits.api import (HasTraits, Instance, Button, ListStr, CInt, Int,
                        List, Dict)
from traitsui.api import (View, Item, ButtonEditor, VGroup, HGroup,
                          Group, TabularEditor)
from traitsui.tabular_adapter import TabularAdapter
from chaco.api import Plot, ArrayPlotData, VPlotContainer, gray
from enable.component_editor import ComponentEditor

# Local imports
from galaxyzoo.processing.api import (TRAINING_IMAGES_DIR, TRAINING_SOLN_PATH,
                                      get_cleared_binary_image,
                                      get_region_bounds, rotate_largest_region,
                                      get_bbox_center, get_largest_region,
                                      point_rotate, crop_around_centroid)


################################################################################
# `DataFrameAdapter` class
################################################################################
class DataFrameAdapter(TabularAdapter):
    """
    TabularAdapter subclass for a pandas dataframe.
    """

    # The DataFrame to be displayed in the UI
    data = Instance(pd.DataFrame, ())

    # Labels for columns of the UI display, by default these are the same as
    # dataframe.columns
    columns = List

    # Number of header rows in the file.
    no_headers = Int

    can_edit = False

    def get_item(self, object, trait, row):
        '''
        Pandas dataframes are accessed differently than numpy arrays.
        Overwritten to support indexing of pandas DataFrames.
        '''
        try:
            return list(getattr(self.object, trait).ix[row])
        except:
            return None


    def get_row_label(self, section, obj=None):
        """
        Returns the label for each row. In this case it is the index.
        """
        ind = self.data.index[section]
        return str(ind)

    def get_text(self, object, trait, row, column):
        """
        Returns the text to display for a specified object.trais[row].column 
        item. Overwritten to return strings corresponding to every item in a 
        pandas DataFrame.
        """
        df = getattr(object, trait)
        col = df.columns[column]
        index = df.index[row]
        item = df[col][index]
        # Round off high precision numbers for display so as to avoid displaying
        # an ellipsis in the table.
        try:
            d = float(item)
            return str(np.round(d, 3))
        except ValueError:
            return item

    def get_width(self, object, trait, column):
        return 50

    def _columns_default(self):
        return [col[5:] for col in self.data.columns]


################################################################################
# `DataInspector` class
################################################################################
class DataInspector(HasTraits):
    """ Data Inspector for the Galaxy problem. """

    # The main container
    image = Instance(VPlotContainer)

    # The ArrayPlotData instance corresponding to the galaxy image
    image_data = Instance(ArrayPlotData)

    # The ArrayPlotData instance corresponding to the orientation fit.
    image_orientation = Instance(ArrayPlotData)

    # The ArrayPlotData instance corresponding to the galaxy pdf
    soln_data = Instance(ArrayPlotData)

    # Button for triggering the next image
    button = Button()

    # List of training jpg image files
    files = ListStr

    # Galaxy ID of the current image
    data_id = CInt

    # DataFrame containing `solutions_training.csv`
    solutions = Instance(pd.DataFrame)

    # Image histogram data for the current image
    histogram_plotdata = Instance(ArrayPlotData)

    # Histogram of the current image
    histogram_plot = Instance(Plot)

    # Tabular adapter required to display the soluions data in tabular format.
    adapter = Instance(TabularAdapter)

    # ArrayPlotData instance associated with the labeled image regions
    labeled_image_data = Instance(ArrayPlotData)

    # The plot showing the labeled image regions
    labeled_image_plot = Instance(Plot)

    # ArrayPlotData instance associated with the cropped image
    cropped_image_data = Instance(ArrayPlotData)

    # The plot showing the cropped image
    cropped_image = Instance(Plot)

    # The currently selected row
    selected_row = Int

    # A cache for the image plots
    image_plot_cache = Dict

    def default_traits_view(self):
        view = View(HGroup(
            Item('image', editor=ComponentEditor(),
                 show_label=False, width=500),
            VGroup(
                Group(
                    Item('labeled_image_plot', editor=ComponentEditor(),
                         show_label=False, dock="tab",
                         full_size=False, padding=0),
                    Item('histogram_plot', editor=ComponentEditor(),
                         show_label=False, dock="tab",
                         full_size=False, padding=0),
                    Item('solutions',
                         editor=TabularEditor(
                             adapter=self.adapter,
                             auto_resize=False,
                             auto_resize_rows=False,
                             stretch_last_section=False,
                             show_row_titles=True,
                             horizontal_lines=False,
                             multi_select=False,
                             drag_move=False,
                             scroll_to_row="selected_row",
                             scroll_to_row_hint="visible",
                             selected_row='selected_row'),
                         show_label=False, dock="tab",
                         full_size=False),
                    label="Solutions", springy=True,
                    layout="tabbed", scrollable=True),
                HGroup(
                    Item('data_id', label='Galaxy ID'),
                    Item('button', label="next",
                         editor=ButtonEditor(), show_label=False)
                )
            )
        ), resizable=True)
        return view

    def get_random_image(self):
        """ Randomly select a data id.
        """
        ind = np.random.randint(0, len(self.files))
        filename = os.path.basename(self.files[ind])
        return int(filename.split('.')[0])

    def get_data_by_id(self, id):
        """ Return the image and the target pdf for the give ID.
        """
        try:
            pdf = self.solutions.ix[id]
            if self.image_plot_cache.has_key(id):
                return pdf, self.image_plot_cache.get(id)
            else:
                filename = os.path.join(TRAINING_IMAGES_DIR, str(id) + '.jpg')
                im = imread(filename)
                self.image_plot_cache[id] = im
            return pdf, im
        except:
            pass

    def update_cropped_image_data(self):
        x = self.image_data.arrays.get('im')[:, :, 0]
        org_region = get_largest_region(x)
        rr1, cc1 = get_bbox_center(org_region.bbox)
        rotated_x = rotate_largest_region(x)
        _x = cc1 - x.shape[1] / 2
        _y = rr1 - x.shape[0] / 2
        cc2, rr2 = point_rotate(_x, _y, -org_region.orientation)
        rr2 += rotated_x.shape[0] / 2
        cc2 += rotated_x.shape[1] / 2
        cropped = crop_around_centroid(rotated_x, rr2, cc2)
        self.cropped_image_data.set_data('im', cropped)

    def update_labeled_image_data(self):
        im = self.image_data.arrays.get('im')[:, :, 0]
        bw = get_cleared_binary_image(im)
        bx, by = get_region_bounds(bw)
        apd = ArrayPlotData(im=bw)
        for i in range(len(bx)):
            minc, maxc = bx[i]
            minr, maxr = by[i]
            key = "bb" + str(i)
            apd.update({key + 'topx': (minc, maxc),
                        key + 'topy': (maxr, maxr),
                        key + 'botx': (minc, maxc),
                        key + 'boty': (minr, minr),
                        key + 'leftx': (minc, minc),
                        key + 'lefty': (minr, maxr),
                        key + 'rightx': (maxc, maxc),
                        key + 'righty': (minr, maxr)})
        plot = Plot(apd)
        plot.img_plot('im', colormap=gray, origin='bottom left')
        for i in range((len(apd.arrays) - 1) / 8):
            key = 'bb' + str(i)
            plot.plot((key + 'topx', key + 'topy'), color='green',
                      line_width=3)
            plot.plot((key + 'botx', key + 'boty'), color='green',
                      line_width=3)
            plot.plot((key + 'leftx', key + 'lefty'), color='green',
                      line_width=3)
            plot.plot((key + 'rightx', key + 'righty'), color='green',
                      line_width=3)
        self.labeled_image_plot = plot

    # Trait initializers ######################################################
    def _files_default(self):
        return [os.path.join(TRAINING_IMAGES_DIR, image) for image in
                os.listdir(TRAINING_IMAGES_DIR)]

    def _solutions_default(self):
        return pd.read_csv(TRAINING_SOLN_PATH, index_col=0)

    def _adapter_default(self):
        return DataFrameAdapter(data=self.solutions)

    def _data_id_default(self):
        return self.get_random_image()

    def _soln_data_default(self):
        soln = self.solutions.ix[self.data_id]
        apd = ArrayPlotData(x=np.arange(self.solutions.shape[1]),
                            y=soln.values)
        return apd

    def _image_data_default(self):
        filename = os.path.join(TRAINING_IMAGES_DIR, str(self.data_id) + '.jpg')
        im = imread(filename)
        self.image_plot_cache[id] = im
        apd = ArrayPlotData(im=im)
        return apd

    def _histogram_plotdata_default(self):
        im = self.image_data.arrays['im']
        hist = histogram(im[:, :, 0])
        apd = ArrayPlotData(x=hist[1], y=hist[0])
        return apd

    def _image_default(self):
        container = VPlotContainer()
        soln_plot = Plot(self.soln_data)
        soln_plot.plot(("x", "y"), type='bar', fill_color='green',
                       bar_width=0.8)
        container.add(soln_plot)
        galaxy_plot = Plot(self.image_data)
        galaxy_plot.img_plot('im')
        container.add(galaxy_plot)
        container.add(self.cropped_image)
        return container

    def _labeled_image_data_default(self):
        im = self.image_data.arrays.get('im')[:, :, 0]
        bw = get_cleared_binary_image(im)
        bx, by = get_region_bounds(bw)
        apd = ArrayPlotData(im=bw)
        for i in range(len(bx)):
            minc, maxc = bx[i]
            minr, maxr = by[i]
            key = "bb" + str(i)
            apd.update({key + 'topx': (minc, maxc),
                        key + 'topy': (maxr, maxr),
                        key + 'botx': (minc, maxc),
                        key + 'boty': (minr, minr),
                        key + 'leftx': (minc, minc),
                        key + 'lefty': (minr, maxr),
                        key + 'rightx': (maxc, maxc),
                        key + 'righty': (minr, maxr)})
        return apd

    def _labeled_image_plot_default(self):
        plot = Plot(self.labeled_image_data)
        plot.img_plot('im', colormap=gray, origin='bottom left')
        for i in range((len(self.labeled_image_data.arrays) - 1) / 8):
            key = 'bb' + str(i)
            plot.plot((key + 'topx', key + 'topy'), color='green',
                      line_width=3)
            plot.plot((key + 'botx', key + 'boty'), color='green',
                      line_width=3)
            plot.plot((key + 'leftx', key + 'lefty'), color='green',
                      line_width=3)
            plot.plot((key + 'rightx', key + 'righty'), color='green',
                      line_width=3)
        return plot

    def _cropped_image_data_default(self):
        x = self.image_data.arrays.get('im')[:, :, 0]
        org_region = get_largest_region(x)
        rr1, cc1 = get_bbox_center(org_region.bbox)
        rotated_x = rotate_largest_region(x)
        _x = cc1 - x.shape[1] / 2
        _y = rr1 - x.shape[0] / 2
        cc2, rr2 = point_rotate(_x, _y, -org_region.orientation)
        rr2 += rotated_x.shape[0] / 2
        cc2 += rotated_x.shape[1] / 2
        cropped = crop_around_centroid(rotated_x, rr2, cc2)
        apd = ArrayPlotData(im=cropped)
        return apd

    def _cropped_image_default(self):
        plot = Plot(self.cropped_image_data)
        plot.img_plot('im', colormap=gray)
        return plot

    # Trait change handlers ###################################################
    def _button_fired(self):
        ind = np.random.randint(0, self.solutions.shape[0])
        self.data_id = self.solutions.index[ind]

    def _data_id_changed(self):
        indices = self.solutions.index.tolist()
        try:
            self.selected_row = indices.index(self.data_id)
        except ValueError:
            pass
        data = self.get_data_by_id(self.data_id)
        if data is not None:
            pdf, im = data
            self.soln_data.set_data('y', pdf.values)
            self.image_data.set_data('im', im)
            hist = histogram(im[:, :, 0])
            self.histogram_plotdata.set_data('x', hist[1])
            self.histogram_plotdata.set_data('y', hist[0])
            self.update_labeled_image_data()
            self.update_cropped_image_data()

    def _selected_row_changed(self, new):
        self.data_id = self.solutions.index[new]


################################################################################
# `ProcessedDataInspector` class
################################################################################
class ProcessedDataInspector(HasTraits):
    """
    Data inspector to inspect specific image IDs, specified by the indices
    trait
    """

    processed_img_dir = "/Users/jaidevd/GitHub/kaggle/galaxyzoo" + \
                        "/new_processed_images"

    indices = List

    current_index = Int

    next_im = Button

    prev_im = Button

    plotdata = Instance(ArrayPlotData)

    plot = Instance(Plot)

    next_condt = 'indices.index(current_index)<len(indices)-1'

    prev_condt = 'indices.index(current_index)!=0'

    def default_traits_view(self):
        view = View(VGroup(
                           Item('plot', editor=ComponentEditor(),
                                show_label=False),
                           HGroup(
                               Item('prev_im',label="Previous",
                                    enabled_when=self.prev_condt,
                                    show_label=False, editor=ButtonEditor()),
                               Item('next_im',label='Next',
                                    enabled_when=self.next_condt,
                                    show_label=False, editor=ButtonEditor())
                           )
                    ), resizable=True)
        return view

    def _current_index_default(self):
        return self.indices[0]

    def _plotdata_default(self):
        impath = os.path.join(self.processed_img_dir,
                              str(self.current_index)+'.png')
        im = imread(impath)
        apd = ArrayPlotData(im=im)
        return apd

    def _plot_default(self):
        plot = Plot(self.plotdata)
        plot.img_plot("im", colormap=gray)
        return plot

    def _current_index_changed(self):
        impath = os.path.join(self.processed_img_dir,
                              str(self.current_index)+'.png')
        im = imread(impath)
        self.plotdata.set_data('im',im)

    def _next_im_fired(self):
        self.current_index = self.indices[self.indices.index(self.current_index) + 1]

    def _prev_im_fired(self):
        self.current_index = self.indices[self.indices.index(self.current_index) - 1]


if __name__ == "__main__":
    ste = DataInspector()
    ste.configure_traits()
