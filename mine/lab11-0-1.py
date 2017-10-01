#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

matplotlib.pyplot.ion()

sess = tf.InteractiveSession()
image = np.array(
    [[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=np.float32)

'''
    image.shape (1, 3, 3, 1)
    image : image
    1 : image의 개수
    3, 3 : 3 x 3 의 크기
    1 : 칼라 크기
'''

plt.imshow(image.reshape(3, 3), cmap='Greys')

'''
    weight.shape (2, 2, 1, 1)
    weight : filter
    2, 2 : 2 x 2 의 크기
    1 : 칼라 크기
    1 : filter를 몇 개 사용할 것인가
'''
weight = tf.constant([[[[1., 10., -1.]], [[1., 10., -1.]]],[[[1., 10., -1. ]], [[1., 10., -1.]]]])

print("image.shape", image.shape)
print("weight.shape", weight.shape)
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
conv2d_img = conv2d.eval()

print("conv2d_img.shape", conv2d_img.shape)
print("conv2d_img", conv2d_img)
