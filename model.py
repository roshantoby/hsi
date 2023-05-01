import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

samples_labels = np.load('data/samples.npy')

samples = samples_labels[:, :-1]
labels = samples_labels[:, -1]

x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=0.4)

rf = RandomForestClassifier()

rf.fit(x_train, y_train)
# TODO Model needs to be trained with data from multiple samples to generalise well.
print(rf.feature_importances_)
print('done')


