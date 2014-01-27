import galaxyzoo.processing.api as gp
from sklearn.decomposition import TruncatedSVD
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

FIRST3 = "/Users/jaidevd/GitHub/kaggle/galaxyzoo/first_three.hdf"

keys=['smooth_inds','feature_inds','artifact_inds','smooth_images',
      'feature_images','artifact_images']
keys = ['/'+key for key in keys]

arrays = gp.get_hdf(FIRST3, keys)

smooth, feature, artifact = arrays['/smooth_images'],  arrays['/feature_images'], arrays['/artifact_images']
tsvd = TruncatedSVD(n_components=3)

smooth_red = tsvd.fit_transform(smooth)
feature_red = tsvd.fit_transform(feature)
artifact_red = tsvd.fit_transform(artifact)
reductions = [smooth_red, feature_red, artifact_red]

# Plotting the three classes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cols = ['r^', 'g^', 'b^']
for x in reductions:
    ax.scatter(x[:,0], x[:,1], x[:,2], cols[reductions.index(x)])
plt.show()
