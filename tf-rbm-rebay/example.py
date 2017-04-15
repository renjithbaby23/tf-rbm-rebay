import numpy as np
import matplotlib.pyplot as plt
from gbrbm import GBRBM
from bbrbm import BBRBM
from tensorflow.examples.tutorials.mnist import input_data
import pandas

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_images = mnist.train.images

n_visible = 784
n_hidden = 64

# gbrbm = GBRBM(n_visible=784, n_hidden=64, learning_rate=0.01,
# momentum=0.95, use_tqdm=True, sample_visible=True, sigma=1)
bbrbm = BBRBM(n_visible=784, n_hidden=64, learning_rate=0.01, momentum=0.95, use_tqdm=True, sample_visible=True, sigma=1)
errs = bbrbm.fit(mnist_images, n_epoches=10, batch_size=10)
# errs = gbrbm.fit(mnist_images, n_epoches=10, batch_size=10)
plt.figure(1)
plt.plot(pandas.rolling_mean(errs, 100))
plt.show()


IMAGE = 1


def show_digit(x):
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    # plt.show()

image = mnist_images[IMAGE]
image_rec = bbrbm.reconstruct(image.reshape(1,-1))

plt.figure(2)
plt.subplot(2, 1, 1)
show_digit(image)
plt.subplot(2, 1, 2)
show_digit(image_rec)
plt.show()

rbm = BBRBM(n_visible, n_hidden, learning_rate=0.01, momentum=0.95,
            err_function='rmse', use_tqdm=False)
#
# rbm = GBRBM(n_visible, n_hidden, learning_rate=0.01, momentum=0.95,
# err_function='rmse', use_tqdm=False, sample_visible=False, sigma=1)
#
rbm.fit(data_x, n_epoches=10, batch_size=10, shuffle=True, verbose=True)

rbm.partial_fit(batch_x)
rbm.reconstruct(batch_x)


rbm.transform(batch_x)
rbm.transform_inv(batch_y)

rbm.get_err(batch_x)
rbm.get_weights()


rbm.set_weights(w, visible_bias, hidden_bias)
rbm.save_weights(filename, name)
rbm.load_weights(filename, name)

