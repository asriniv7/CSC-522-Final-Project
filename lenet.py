class FullyConnectedLayer(object):

	def __init__ (self, n_in, n_out, activation_fn = sigmoid, p_dropout = 0.0):
		self.n_in = n_in
		self.n_out = n_out
		self.activation_fn = activation_fn
		self.p_dropout = p_dropout

		#Initialize weights and biases
		self.w = theano.shared(
			np.asarray(
				np.random.normal(
					loc = 0.0, scale = np.sqrt(1.0/n_out), size = (n_in,n_out)),
				dtype = theano.config.floatX),
			name = 'w', borrow = True)

		self.b = theano.shared(
			np.asarray(np.random.normal(loc = 0.0, scale = 1.0, size =(n_out,)),
				dtype = theano.config.floatX),
			name = 'b', borrow = True)

		self.params = [self.w, self.b]

	def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
		self.inpt = inpt.reshape((mini_batch_size, self.n_in))
		self.output = self.activation_fn(
			(1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
		self.y_out = T.argmax(self.output, axis = 1)
		self.inpt_dropout = dropout_layer(
			inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
		self.output_dropout = self.activation_fn(
			T.dot(self.input_dropout, self.w)+ self.b)

	def accuracy(self, y):
		return T.mean(T.eq(y, self.y_out))


##########################################

class ConvPoolLayer(object):
	#Used to create a combination of convolution and maxpool layers

	def __init__ (self, filter_shape, image_shape, poolsize = (2,2), activation_fn = sigmoid):

		#filter_shape is a tuple of length 4 : number of filters, number of input feautre maps,
		#filter height, and width  

		self.fiter_shape = filter_shape
		self.image_shape = image_shape
		self.poolsize = poolsize
		self.activation_fn = activation_fn

		#initialize weights and biases
		n_out = (filter_shape[0]* np.prod(filter_shape[2:])/np.prod(poolsize))

		self.w = theano.shared(
			np.asarray(
				np.random.normal(loc = 0, scale = np.sqrt(1.0/n_out), size = filter_shape),
				dtype = theano.config.floatX),
			borrow = True)

		self.b = theano.shared(
			np.asarray(
				np.random.normal(loc = 0, scale = 1.0, size = (filter_shape[0],)),
				dtype = theano.config.floatX),
			borrow = True)

		self.params = [self.w, self.b]

	def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
		self.inpt = inpt.reshape(self.image_shape)
		conv_out = conv.conv2d(
			input = self.inpt, filters = self.w, filter_shape = self. filter_shape,
			image_shape = self.image_shape)

		pooled_out = downsample.max_pool_2d(
			input = conv_out, ds = self.poolsize, ignore_border = True)

		self.output = self.activation_fn(
			pooled_out + self.b.dimshuffle('x',0,'x','x'))

		self.output_dropout = self.output #as there's no droput in the convlayer


###########################################

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))
