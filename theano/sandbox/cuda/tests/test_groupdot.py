from nose.plugins.skip import SkipTest
import numpy 
import numpy as np

import theano
from theano.gof.python25 import any
import theano.tensor as T
import theano.tests.unittest_tools as utt

# Skip test if cuda_ndarray is not available.
#import theano.sandbox.cuda as cuda
#if cuda.cuda_available == False:
#    raise SkipTest('Optional package cuda disabled')
theano.config.exception_verbosity= 'high'
theano.config.profile= True
if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode(
        'FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')


def test_groupdot():
    x = T.fmatrix('x')
    w = T.tensor3('w',dtype='float32')
    b = T.fmatrix('b')
    c = T.vector('c',dtype='int32')
    z = T.nnet.GroupDot(51)(x, w,b,c)
    
    n_batch =100
    n_hid=256
    n_clust=51
    n_classes=10000
        
    f = theano.function([x,w,b,c], z, mode=mode_without_gpu, name='my function')
    f_gpu = theano.function([x,w,b,c], z, mode=mode_with_gpu)                              
    
    x = numpy.arange(n_batch * n_hid, dtype='float32').reshape(n_batch, n_hid)
    w = np.random.rand(n_clust,n_hid,n_classes).astype('float32')
    b = np.random.rand(n_clust, n_classes).astype('float32')
    c = np.random.randint(0, n_clust, size=(n_batch,)).astype('int32')
    out=f(x,w,b,c)
    gout=f_gpu(x,w,b,c)
    assert numpy.allclose(out, gout), numpy.absolute(out - gout)
    