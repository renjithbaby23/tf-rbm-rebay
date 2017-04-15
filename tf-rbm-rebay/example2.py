import numpy as np
import matplotlib.pyplot as plt
from gbrbm import GBRBM
from bbrbm import BBRBM
from tensorflow.examples.tutorials.mnist import input_data
import pandas
from sklearn import datasets as d


# ##########################################################################################
def show_plot(x, in_title='Title'):
    plt.scatter(x[:, 0], x[:, 1])
    plt.title(in_title)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.show()

# ##########################################################################################
# Defining parameters
n_visible = 2
n_hidden = 5
n_train = 1000
n_test = 200

n_epoches = 10
batch_size = 100

# ##########################################################################################
# defining input data

# # MNIST image
# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
# X = mnist.train.images
#
# # generating randomn data
# X = np.random.uniform(0, 1, (n_train, n_visible))
#
# # two cluster data
# X = np.array([], float)
# X1 = np.random.uniform(0.75, 1, (int(n_train / 2), n_visible))
# X2 = np.random.uniform(0.25, 0.5, (int(n_train / 2), n_visible))
# X = np.vstack([X, X1]) if X.size else X1
# X = np.vstack([X, X2]) if X.size else X2

# Generating half moon data
X = d.make_moons(n_samples=n_train, shuffle=True, noise=0.05, random_state=0)
X = np.array(X[0], dtype=float)

# ##########################################################################################
# Running RBM
bbrbm = BBRBM(n_visible=n_visible, n_hidden=n_hidden, learning_rate=0.01, momentum=0.95, use_tqdm=True)
gbrbm = GBRBM(n_visible=n_visible, n_hidden=n_hidden, learning_rate=0.01, momentum=0.95, use_tqdm=True, sample_visible=False, sigma=1)
errs1 = bbrbm.fit(X, n_epoches=n_epoches, batch_size=batch_size)
errs2 = gbrbm.fit(X, n_epoches=n_epoches, batch_size=batch_size)

# ##########################################################################################
# Ploting the errors
plt.figure('Errors')
plt.subplot(2,1,1)
plt.plot(pandas.rolling_mean(errs1, 50))
plt.title('BBRBM errors')
plt.subplot(2,1,2)
plt.plot(pandas.rolling_mean(errs2, 50))
plt.title('GBRBM errors')
# plt.show()

# ##########################################################################################
# Ploting the input and reconstructed input

train_rec1 = bbrbm.reconstruct(X)

plt.figure('Data Visualisation')
plt.subplot(3, 2, 1)
show_plot(X, 'train data')
plt.subplot(3, 2, 2)
show_plot(train_rec1, 'BBRBM reconstructed train data')
train_rec2 = gbrbm.reconstruct(X)
plt.subplot(3, 2, 3)
show_plot(train_rec2, 'GBRBM reconstructed train data')

# ##########################################################################################
# Ploting the test and its reconstructed version

# # Moon data for testing
# test = d.make_moons(n_samples=n_test, shuffle=True, noise=0.5, random_state=0)
# test = np.array(test[0], dtype=float)

# Normal distribution data
test = np.random.multivariate_normal([.5, .5], [[.051, 0], [0, 1]], n_test)
print('shape of test: ', np.shape(test[0]))

# # two cluster data
# test = np.array([], float)
# t1 = np.random.uniform(0.5, 1, (int(n_test / 2), n_visible))
# t2 = np.random.uniform(0, 0.5, (int(n_test / 2), n_visible))
# test = np.vstack([test, t1]) if test.size else t1
# test = np.vstack([test, t2]) if test.size else t2

plt.subplot(3, 2, 4)
show_plot(test, 'test data')
test_rec1 = bbrbm.reconstruct(test)
plt.subplot(3, 2, 5)
show_plot(test_rec1, 'BBRM reconstructed test data')
test_rec2 = gbrbm.reconstruct(test)
plt.subplot(3, 2, 6)
show_plot(test_rec2, 'GBRM reconstructed test data')
plt.show()
