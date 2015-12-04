import network3
from network3 import Network 
from network3 import SoftmaxLayer, FullyConnectedLayer, ConvPoolLayer
import winsound 



mini_batch_size = 10

f = FullyConnectedLayer(n_in = 20 * 12 * 12, n_out = 100)
s = SoftmaxLayer(n_in = 100, n_out = 25)
c = ConvPoolLayer(image_shape = (mini_batch_size, 1, 28, 28),
	filter_shape = (20, 1, 5, 5), 
	poolsize = (2,2))

net = Network([c, f ,s], mini_batch_size)

print "\n Network Created. Now Loading Data. Please be patient. \n"

train, valid, test = network3.load_data_shared()

print "\n \n \nData Sucessfully Loaded \n"
#winsound.Beep(2100, 1000) 

net.SGD(train, 30, mini_batch_size, 0.1, valid, test) 