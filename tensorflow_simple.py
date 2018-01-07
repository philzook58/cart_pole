import tensorflow as tf

#tensorboard --logdir="./graphs"

x = tf.placeholder(tf.float32, [1,3])

#W = tf.Variable(tf.zeros([2, 3]))
W = tf.get_variable("W", dtype=tf.float32, 
  initializer=tf.constant([[1.0, 0.0, 0.0],[3.0,4.0, 5.0]]))
#b = tf.Variable(tf.zeros([10]))

#tf.matmul(x, W)
y = tf.matmul(W,tf.transpose(x))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
grad = tf.gradients(y, x)
#optimizer = tf.train.GradientDescentOptimizer(0.5)
optimizer = tf.train.AdamOptimizer(learning_rate=0.5)
train = optimizer.minimize(y)
#x = tf.placeholder(tf.float32, shape=[3])
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	print(sess.run(y, {x: [[1.0, 2.0, 3.0]]}))
	print(sess.run(grad, {x: [[1.0, 0.0, 0.0]]})) 
	for step in range(10):  
		sess.run(train, {x: [[1.0, 0.0, 0.0]]})


writer.close()
"""
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
            """