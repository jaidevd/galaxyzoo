from traits.api import HasTraits, Instance, Button, List, Unicode
from traitsui.api import View, Item, ButtonEditor, HGroup, VGroup
from enable.component_editor import ComponentEditor
from chaco.api import Plot, ArrayPlotData
from matplotlib.pyplot import imread
import numpy as np
import pandas as pd
import os
import json

ROOT = "/Users/jaidevd/GitHub/kaggle/galaxyzoo"
DEFECTS = os.path.join(ROOT,'defective_files.json')
SOLN_PDF = os.path.join(ROOT, 'solutions_training.csv')
TRAIN_DATA = os.path.join(ROOT, 'images_training')


class Viewer(HasTraits):

    img_list = List(Unicode)
    
    current_image = Unicode
    
    soln_data = Instance(pd.DataFrame)
    
    next_im = Button()
    
    prev_im = Button()
    
    plotdata = Instance(ArrayPlotData)
    
    histdata = Instance(ArrayPlotData)
    
    plot = Instance(Plot)
    
    histplot = Instance(Plot)
    
    condition = 'img_list.index(current_image)<len(img_list)-1'
    
    prev_condition = 'img_list.index(current_image)!=0'
    
    def default_traits_view(self):
        view = View(
                    VGroup(
                    HGroup(Item('plot', editor=ComponentEditor(),
                            show_label=False),
                           Item('histplot', editor=ComponentEditor(),
                            show_label=False)),
                    HGroup(Item('prev_im', editor=ButtonEditor(),
                            enabled_when=self.prev_condition,label="previous",
                            show_label=False),
                           Item('next_im', editor=ButtonEditor(),
                            enabled_when=self.condition, label="next",
                            show_label=False),
                           Item('current_image', show_label=False))),
                    resizable=True)
                    
        return view
    
    def _img_list_default(self):
        with open(DEFECTS,'r') as f:
            defects = json.load(f)
        final = [f[0] for f in defects]
        return final

    def _current_image_default(self):
        return self.img_list[0]
    
    def _soln_data_default(self):
        df = pd.read_csv(SOLN_PDF,index_col=0)
        indices = [int(im.split('.')[0]) for im in self.img_list]
        return df.ix[indices]
    
    def _plotdata_default(self):
        try:
            x = imread(os.path.join(TRAIN_DATA,self.current_image))
        except:
            x = np.zeros((424,424),dtype=np.uint8)
        apd = ArrayPlotData(im=x)
        return apd
    
    def _histdata_default(self):
        x = self.soln_data.ix[int(self.current_image.split('.')[0])].values
        apd = ArrayPlotData(x=np.arange(self.soln_data.shape[1]),y=x)
        return apd
    
    def _histplot_default(self):
        plot = Plot(self.histdata)
        plot.plot(("x","y"), type='bar', fill_color='green', bar_width=.5)
        return plot
        
    def _plot_default(self):
        plot = Plot(self.plotdata)
        plot.img_plot('im')
        return plot
    
    def _next_im_fired(self):
        current_index = self.img_list.index(self.current_image)
        self.current_image = self.img_list[current_index + 1]
    
    def _prev_im_fired(self):
        current_index = self.img_list.index(self.current_image)
        self.current_image = self.img_list[current_index - 1]
    
    def _current_image_changed(self):
        try:
            x = imread(os.path.join(TRAIN_DATA,self.current_image))[:,:,0]
            self.plotdata.set_data('im', x)
        except:
            pass
        y = self.soln_data.ix[int(self.current_image.split('.')[0])].values
        self.histdata.set_data('y',y)


if __name__ == "__main__":
    Viewer().configure_traits()