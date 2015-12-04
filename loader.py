from PIL import Image
import os
from os.path import isfile,join
import numpy as np
import cPickle as pickle 
import gc
import random

gc.enable()

#create lists for training, validation, and testing data
train_data = []
train_class = []
valid_data = []
valid_class = []
test_data = []
test_class = []


def vectorized_result(j):
	e = np.zeros((50, 1))
	e[j-1] = 1.0
	return e

for root, dirs, files in os.walk('../Datasets/Objects'):
	print "\n"

	for f in files:
		if f.endswith('.jpg') and int(f[:3]) < 25:
			path = join(root,f)

			im = Image.open(path)#.convert('LA')

			ob = int(f[:3])
			a = np.asarray(im)
			im.close()
			a = a[:,:,0]
			a = np.reshape(a, (256*256))
			a = a/256 

			#result = vectorized_result(ob)

			#data = a, result

			r = random.randint(1,10)

			if r < 6:
				train_data.append(a) # append the image as an array
				train_class.append(ob) #append the image class to the list
			elif r < 8:
				valid_data.append(a) 
				valid_class.append(ob)
			else:
				test_data.append(a)
				test_class.append(ob) 			

	print root, " \n  ....... done!! \n"

train_array = np.array(train_data, dtype = np.float32)
valid_array = np.array(valid_data, dtype = np.float32)
test_array = np.array(test_data, dtype = np.float32) 

tr_class_array = np.array(train_class, dtype = np.int64)
v_class_array = np.array(valid_class, dtype = np.int64)
te_class_array = np.array(test_class, dtype = np.int64) 

train = train_array, tr_class_array
valid = valid_array, v_class_array
test = test_array, te_class_array

print "Train 0 is of type ", type(train[0]), "\n"
print "Shape of Train 0 is ", train[0].shape, "\n" 

print "Train = ", len(train), " \t Valid =  ", len(valid), " \t Test = ", len(test), "\n"

all_data = train, valid, test

print "All data created! \n"

#pickle.dump(train, open("cal_data/train.p", 'wb'))

#print "Train pickled"
#pickle.dump(valid, open("cal_data/valid.p", 'wb'))

#print "Valid pickled"
#pickle.dump(test, open("cal_data/test.p", 'wb'))

#print "Test pickled" 

pickle.dump(all_data, open("cal_data/all_data.p", "wb"))
print "All Data Pickled!"


