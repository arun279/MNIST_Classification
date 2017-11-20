#DNN using MNIST data
import tensorflow as tf

"""
input > weight  > hidden layer 1 (activation function)
> weights > hidden l 2 (activation function) 
> weights > output layer

compare output to intended output >
cost function (cross entropy)
optimization function ( optimizer) > minimize cost
(AdamOptimizer...SGD, AdaGrad)

backpropagation
feed forward + backprop

"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp/data/",one_hot=True)

n_nodes_hl1 = 50
n_nodes_hl2 = 5
n_nodes_hl3 = 5

n_classes = 10
batch_size = 100
x = tf.placeholder('float',[None, 784]) #flatten to 1 x 784
y = tf.placeholder('float') #Labels

def NN_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.Variable(tf.random_normal([784,n_nodes_hl1]))),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
	#hidden_2_layer = {'weights':tf.Variable(tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2]))),
	#				  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
	#hidden_3_layer = {'weights':tf.Variable(tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3]))),
	#				  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer   = {'weights':tf.Variable(tf.Variable(tf.random_normal([n_nodes_hl1,n_classes]))),
					  'biases': tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)


	#l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
	#l2 = tf.nn.relu(l2)


	#l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
	#l3 = tf.nn.relu(l3)


	output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x):
	prediction = NN_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y))

	#Minimize cost
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	epochs = 100

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				current_x,current_y = mnist.train.next_batch(batch_size)
				_,c = sess.run([optimizer,cost], feed_dict = {x:current_x,y:current_y})
				epoch_loss += c
				print('Epoch', epoch, 'done out of', epochs,'loss:',epoch_loss)
			correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
			accuracy = tf.reduce_mean(tf.cast(correct,'float'))
			print('Validation accuracy:',accuracy.eval({x:mnist.validation.images, y: mnist.validation.labels}))
			print('Test accuracy:',accuracy.eval({x:mnist.test.images, y: mnist.test.labels}))
train_neural_network(x)