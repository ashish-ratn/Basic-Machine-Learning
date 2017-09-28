import tensorflow as tf

W = tf.Variable([.3],dtype=tf.float32)
b = tf.Variable([.3],dtype=tf.float32)

x = tf.placeholder(tf.float32)
linear_model = W*x+b
y = tf.placeholder(tf.float32)

loss = tf.reduce_sum(tf.square(linear_model-y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

x_train = [4,5,3,5,4]
y_train = [9,4,-2,3,5]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
	sess.run(train, {x:x_train, y:y_train})

	curr_W, curr_b,curr_loss = sess.run([W,b,loss],{x:x_train,y:y_train})
	print("W:%s b: %s loss: %s"%(curr_W,curr_b,curr_loss))
