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

    # m_batch =1000
    # m_hid=512
    # m_clust=100
    # m_classes=10000

    # p_batch =1000
    # p_hid=1024
    # p_clust=1000
    # p_classes=10000
    
    f = theano.function([x,w,b,c], z, mode=mode_without_gpu, name='cpu')
    f_gpu = theano.function([x,w,b,c], z, mode=mode_with_gpu, name='gpu')                              
    

    n_batch=50
    n_hid=300
    n_clust=20
    n_classes=7000
    

    def cmp(n_batch, n_hid,    n_clust,n_classes):
        x = numpy.arange(n_batch * n_hid, dtype='float32').reshape(n_batch, n_hid)
        w = np.random.randint(0,100,size=(n_clust,n_hid,n_classes)).astype('float32')
        b = np.random.randint(0,100,size=(n_clust, n_classes)).astype('float32')
        c = np.random.randint(0, n_clust, size=(n_batch,)).astype('int32')

        output=numpy.zeros(shape=(n_batch, n_classes))
        for i in range(n_batch):
            output[i] = np.dot(x[i,:],w[c[i],:,:])+b[c[i]]
        out=f(x,w,b,c)
        gout=f_gpu(x,w,b,c)
        assert numpy.allclose(out, output), numpy.absolute(out - output)
        assert numpy.allclose(gout, output), numpy.absolute(gout - output)
    
    cmp(50,300,20,7000)
    cmp(100,256,51,10000)
    cmp(1000,512,100,10000)
        
    #x2 = numpy.arange(m_batch * m_hid, dtype='float32').reshape(m_batch, m_hid)
    #w2 = np.random.rand(m_clust,m_hid,m_classes).astype('float32')
    #b2 = np.random.rand(m_clust, m_classes).astype('float32')
    #c2 = np.random.randint(0, m_clust, size=(m_batch,)).astype('int32')

    # x3 = numpy.arange(p_batch * p_hid, dtype='float32').reshape(p_batch, p_hid)
    # w3 = np.random.rand(p_clust,p_hid,p_classes).astype('float32')
    # b3 = np.random.rand(p_clust, p_classes).astype('float32')
    # c3 = np.random.randint(0, p_clust, size=(p_batch,)).astype('int32')

    # print 'x'
    # print x[0,:]
    # print 'w'
    # print w[c[0],:,:]
    # print 'b'
    # print b
    # print 'c'
    # print c[0]
    
    
    
    #print out1
    #out2=f(x2,w2,b2,c2)
    #out3=f(x3,w3,b3,c3)
    
    #gout=f_gpu(x3,w3,b3,c3)
    
    
