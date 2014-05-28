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

""" 1. why so fast
    2. grad?
    3. comparing
"""

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
    #f_gpu = theano.function([x,w,b,c], z, mode=mode_with_gpu)                              
    
    x = numpy.arange(n_batch * n_hid, dtype='float32').reshape(n_batch, n_hid)
    w = np.random.rand(n_clust,n_hid,n_classes).astype('float32')
    b = np.random.rand(n_clust, n_classes).astype('float32')
    c = np.random.randint(0, n_clust, size=(n_batch,)).astype('int32')
    out=f(x,w,b,c)
    #gout=f_gpu(x,w,b,c)



    # def cmp(n_batch, n_hid, n_classes, n_clust):    
    #     x = T.fmatrix('x')
        
    #     z = T.nnet.GroupDot(51)(x,np.random.rand(n_clust,n_hid,n_classes).astype('float32'),
    #         np.random.rand(n_clust, n_classes).astype('float32'),
    #         np.random.randint(0, n_clust, size=(n_batch,)).astype('int32')
    #         )
        
    #     f = theano.function([x],z, mode=mode_without_gpu)
    #     f_gpu = theano.function([x],z,mode=mode_with_gpu)

    #     data = numpy.arange(n_batch * n_hid, dtype='float32').reshape(n_batch, n_hid)

    #     out=f(data)
    #     gout=f_gpu(data)

   










    # assert numpy.allclose(out, gout), numpy.absolute(out - gout)

    # cmp(100,256,10000,51)
    
    # cmp(100,256,10000,51)
    # cmp(100,256,10000,51)
    # cmp(100,256,10000,51)
    # cmp(100,256,10000,51)
    # print 'came here'
    # cmp(10,111,500,35)
#rval = GroupDotGrad(n_groups=self.n_groups)(state_below,
#                                                   matrix, biases,
#                                                     groups, gout)

# def test_groupdot_grad():
#     x = T.fmatrix('x')
#     w = T.tensor3('w',dtype='float32')
#     b = T.fmatrix('b')
#     c = T.vector('c',dtype='int32')
#     gout = 
#     n_clust = T.scalar('n_clust')
#     #z = T.nnet.GroupDot(51)(x, w,b,c)
#     GroupDotGrad(n_clust)(state_below,
#                                                     matrix, biases,
#                                                     groups, gout)
#     #alpha = T.nnet.GroupDotGrad(51)
#     f = theano.function([x,w,b,c], z, mode=mode_without_gpu, name='my function')
#     #f_gpu = theano.function([x], z, mode=mode_with_gpu)                              
#     out = f(numpy.arange(n_batch * n_hid, dtype='float32').reshape(n_batch, n_hid))

    
#     #def cmp(n_batch, n_hid, n_classes, n_clust):
    
#     # n_batch =100
#     # n_hid=256
#     # n_clust=51
#     # n_classes=10000

#     # x = numpy.arange(n_batch * n_hid, dtype='float32').reshape(n_batch, n_hid)
#     # w = np.random.rand(n_clust,n_hid,n_classes).astype('float32')
#     # b = np.random.rand(n_clust, n_classes).astype('float32')
#     # c = np.random.randint(0, n_clust, size=(n_batch,)).astype('int32')
#     # out=f(x,w,b,c)
#     # print out
#     # #print out
#   #      gout = f_gpu(data)
#         #print gout
#         #assert numpy.allclose(out, gout), numpy.absolute(out - gout)

#     #cmp(100,256,10000,51)
    
#     #cmp(100,256,10000,51)
#     #cmp(100,256,10000,51)
#     #cmp(100,256,10000,51)
#     #cmp(100,256,10000,51)
#     #print 'came here'
#     #cmp(10,111,500,35)
