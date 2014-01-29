import galaxyzoo.processing.api as gp
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier)

CLASSIFIERS = [KNeighborsClassifier, LinearSVC, SVC, RandomForestClassifier,
               AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier]

FIRST3 = "/Users/jaidevd/GitHub/kaggle/galaxyzoo/first_three.hdf"

keys=['smooth_inds','feature_inds','artifact_inds','smooth_images',
      'feature_images','artifact_images']
keys = ['/'+key for key in keys]

arrays = gp.get_hdf(FIRST3, keys)

smooth, feature, artifact = arrays['/smooth_images'], arrays['/feature_images'], arrays['/artifact_images']
tsvd = TruncatedSVD(n_components=3)

smooth_red = tsvd.fit_transform(smooth)
feature_red = tsvd.fit_transform(feature)
artifact_red = tsvd.fit_transform(artifact)

# Make a DataFrame from the data

indices = arrays['/smooth_inds'], arrays['/feature_inds'], arrays['/artifact_inds']
indices = np.concatenate((indices[0], indices[1], indices[2]))
data = np.concatenate((smooth_red, feature_red, artifact_red), axis=0)
df = pd.DataFrame(data, index=indices)
print df.shape

# Shuffle the rows
indices = df.index.values.copy()
np.random.shuffle(indices)
df = df.ix[indices]
print df.shape

df['targets'] = np.zeros((df.shape[0],1))
df['targets'].ix[arrays['/feature_inds']] = 1
df['targets'].ix[arrays['/artifact_inds']] = 2

# Split the dataset into training and testing arrays
data = df.values
train = data[:60000,:]
test = data[60000:,:]
test_inds = df.index.values[60000:]

for classifier in CLASSIFIERS:
    if classifier == SVC:
        cls = classifier(probability=True)
    else:
        cls = classifier()
    cls.fit(train[:,:3], train[:,3].ravel())
    if hasattr(cls, 'predict_proba'):
        pp = cls.predict_proba(test[:,:3])
    else:
        pp = cls._predict_proba_lr(test[:,:3])
    org = gp.solutions.ix[test_inds].values[:,:3]
    rmse = gp.rmse(pp, org)
    score = cls.score(test[:,:3], test[:,3].ravel())
    print classifier, '\t', rmse, '\t', score
