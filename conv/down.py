from theano.tensor.signal import downsample
import numpy as np
import pylab
from PIL import Image
import theano
from theano import tensor as T
from theano.tensor.nnet import conv


input = T.dtensor4('input')
maxpool_shape = (2, 2)
pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=True)
f = theano.function([input],pool_out)

im = Image.open('conv 1.png')

print im.size
width, height = im.size   # Get dimensions

left = (width - 500)/2
top = (height - 500)/2
right = (width + 500)/2
bottom = (height + 500)/2

im = im.crop((left, top, right, bottom))

print im.size

#im.show()

a = np.array(im)

print a.shape

b = np.zeros(shape= (3,2,500,500))

b[0,0,:,:] = a[:,:,0]

d = f(b)[0,0,:,:]



print d.shape

c = np.zeros(shape = (3,2,250,250))
c[0,0,:,:] = d

k = f(c)[0,0,:,:]

im_new = Image.fromarray(d).convert('RGB')
im_new.show()
im_new.save('conv_256_1.jpg')