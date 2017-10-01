import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1]) # 헷갈리는구만?

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logit = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logit)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.arg_max(hypothesis, 1) # 왜 1일까 0일것같은데..
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(100000):
		sess.run(optimizer, feed_dict={X: x_data, Y: y_data})

		if step % 200 == 0:
			c, a = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})

			print("Step : {:5}\tCost: {:.3f}\tAccuracy: {:.2}".format(step, c, a))

	pred = sess.run(prediction, feed_dict={X: x_data, Y: y_data})
	for p, y in zip(pred, y_data.flatten()):
		print("[{}] Prediction : {}\tTrue Y: {}".format(p == int(y), p, int(y)))


