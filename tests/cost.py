import theano
import numpy as np
from pylearn2.utils import sharedX
import theano.tensor as T
from pylearn2.format.target_format import OneHotFormatter

def printtypes(input): 
	print type(input),input.type
	return
array_clusters = sharedX(np.array([1,2,3,1,2,1]))
Yes = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]])
print Yes
print Yes.shape

Y = theano.tensor.matrix()
argx = T.argmax(Y, axis=0)
printtypes(argx)
cls = array_clusters[T.cast(argx, 'int32')]

#T.addbroadcast(cls, 1).dimshuffle(0).astype('uint32'))
out = OneHotFormatter(3).theano_expr(cls.astype('uint32'))
#printtypes(cls)
f = theano.function([Y],out)
print f(Yes)
print type(f(Yes))
print f(Yes).shape
