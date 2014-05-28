from nose.plugins.skip import SkipTest
import numpy 
import numpy as np

import theano
from theano.gof.python25 import any
import theano.tensor as T
import theano.tests.unittest_tools as utt

# Skip test if cuda_ndarray is not available.
import theano.sandbox.cuda as cuda
if cuda.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode(
        'FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')


def test_groupdot():
    x = T.fmatrix('x')
    def run(n_batch, n_hid, n_classes, n_clust):
        data = numpy.arange(n_batch * n_hid, dtype='float32').reshape(n_batch, n_hid)
        z = T.nnet.GroupDot(n_clust)(x, np.random.rand(n_clust,n_hid,n_classes).astype('float32'),
                                    np.random.rand(n_clust, n_classes).astype('float32'),
                                    np.random.randint(0, n_clust, size=(n_batch,)))
        #f = theano.function([x], z, mode=mode_without_gpu)
        f_gpu = theano.function([x], z, mode=mode_with_gpu)                              
    
        #out = f(data)
        #print out
        gout = f_gpu(data)
        #print gout
        #assert numpy.allclose(out, gout), numpy.absolute(out - gout)

    #cmp(100,256,10000,51)
    #run(100,256,10000,51)
    run(100,256,10000,100)
    # run(10,111,500,35)
    # run(10,111,500,35)
    # run(10,111,500,35)
