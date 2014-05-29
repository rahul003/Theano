from nose.plugins.skip import SkipTest
import numpy 
import numpy as np
import __builtin__
import theano
from theano.gof.python25 import any
import theano.tensor as T
import theano.tests.unittest_tools as utt

# Skip test if cuda_ndarray is not available.
import theano.sandbox.cuda as cuda
if cuda.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')
theano.config.exception_verbosity= 'high'
#theano.config.profile= True

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

    f = theano.function([x,w,b,c], z, mode=mode_without_gpu, name='cpu')
    f_gpu = theano.function([x,w,b,c], z, mode=mode_with_gpu, name='gpu')                              
    

    n_batch=50
    n_hid=300
    n_clust=20
    n_classes=7000
    
    def cmp(n_batch, n_hid,    n_clust,n_classes):
        x = numpy.arange(n_batch * n_hid, dtype='float32').reshape(n_batch, n_hid)
        w = np.random.rand(n_clust,n_hid,n_classes).astype('float32')
        b = np.random.rand(n_clust, n_classes).astype('float32')
        c = np.random.randint(0, n_clust, size=(n_batch,)).astype('int32')

        output=numpy.zeros(shape=(n_batch, n_classes))
        for i in range(n_batch):
            output[i] = np.dot(x[i,:],w[c[i],:,:])+b[c[i]]
        out=f(x,w,b,c)
        gout=f_gpu(x,w,b,c)
        assert numpy.allclose(out, output)
        assert numpy.allclose(gout, output)
    
    cmp(50,300,20,7000)
    #cmp(100,256,51,10000)
    
    #this fails if gpu mem is less than 2gig
    #cmp(1000,512,100,10000)


# def test_verify_groupdotgrad():
#     x = T.fmatrix('x')
#     w = T.tensor3('w',dtype='float32')
#     b = T.fmatrix('b')
#     c = T.vector('c',dtype='int32')
#     n_clust=20
#     z = T.nnet.GroupDot(n_clust)(x,w,b,c)
#     func = theano.function([x,w,b,c], z, mode=mode_without_gpu, name='cpu')
#     f_gpu = theano.function([x,w,b,c], z, mode=mode_with_gpu, name='gpu')             
#     n_batch=50
#     n_hid=300
#     n_classes=7000
#     x = numpy.arange(n_batch * n_hid, dtype='float32').reshape(n_batch, n_hid)
#     w = np.random.rand(n_clust,n_hid,n_classes).astype('float32')
#     b = np.random.rand(n_clust, n_classes).astype('float32')
#     c = np.random.randint(0, n_clust, size=(n_batch,)).astype('int32')
    
    
    
    
    
#     out=func(x,w,b,c)
#     gout=f_gpu(x,w,b,c)
#     numgrad = numeric_grad(func, [x,w,b,c], 3e-4)


# class numeric_grad(object):
#     """
#     Compute the numeric derivative of a scalar-valued function at a particular
#     point.
#     """
#     type_eps = {'float64': 1e-7,
#                 'float32': 3e-4,
#                 numpy.dtype('float64'): 1e-7,
#                 numpy.dtype('float32'): 3e-4}

#     def __init__(self, f, pt, eps=None, out_type=None):
#         """Return the gradient of f at pt.

#         :param f: a differentiable function such that f(*pt) is a scalar
#         :param pt: an ndarray, a list of ndarrays or tuple of ndarrays
#         :param out_type: dtype of output, if complex (i.e. 'complex32' or
#         'complex64')
#         This function computes the gradient by a one-sided finite
#         differences of a fixed step size (eps).

#         It is assumed that f(...) will return a scalar.
#         It is assumed that all f's inputs are numpy.ndarray objects.

#         :param eps: the stepsize for the finite differencing.  None means
#           input dtype-dependent. See `type_eps`.
#         """

#         def prod(inputs):
#             rval = 1
#             for i in inputs:
#                 rval *= i
#             return rval

#         packed_pt = False
#         if not isinstance(pt, (list, tuple)):
#             pt = [pt]
#             packed_pt = True
#         print type(pt)
#         apt = [numpy.array(p) for p in pt]
#         print type(apt)
        
#         shapes = [p.shape for p in apt]
#         dtypes = [str(p.dtype) for p in apt]

#         # TODO: remove this eventually (why was this here in the first place ?)
#         # In the case of CSM, the arguments are a mixture of floats and
#         # integers...
#         # if not dtypes == [dtypes[0]] * len(apt):
#         #      raise TypeError('All function arguments must have same dtype')

#         total_size = __builtin__.sum(prod(sh) for sh in shapes)

#         #working_dtype = __builtin__.min((self.type_eps[dt], dt)
#                                 #        for dt in dtypes)[1]
#         working_dtype = 'float32'
#         # create un-initialized memory
#         x = numpy.ndarray((total_size,), dtype=working_dtype)
#         if (not out_type is None) and (out_type.startswith('complex')):
#             gx = numpy.ndarray((total_size,), dtype=out_type)
#         else:
#             gx = numpy.ndarray((total_size,), dtype=working_dtype)

#         if eps is None:
#             eps = __builtin__.max(self.type_eps[dt] for dt in dtypes)

#         # set up aliases so that apt[i] is backed by memory in x
#         # and self.gf is backed by memory in gx
#         cur_pos = 0
#         self.gf = []
#         #print x
#         for i, p in enumerate(apt):
#             p_size = prod(p.shape)
#             # set up alias
#             apt[i] = x[cur_pos: cur_pos + p_size].reshape(p.shape)
#             self.gf.append(gx[cur_pos: cur_pos + p_size].reshape(p.shape))
#             # initialize with p's value
#             apt[i][...] = p
#             cur_pos += p_size
        

#         #f_x = f(*[p.copy() for p in apt])
#         f_x = f()


#         # now iterate over the elements of x, and call f on apt.
#         x_copy = x.copy()
#         for i in xrange(total_size):
#             x[:] = x_copy

#             x[i] += eps
#             f_eps = f(*apt)

#             # TODO: remove this when it is clear that the next
#             # replacemement does not pose problems of its own.  It was replaced
#             # for its inability to handle complex variables.
#             # gx[i] = numpy.asarray((f_eps - f_x) / eps)

#             gx[i] = ((f_eps - f_x) / eps)

#         if packed_pt:
#             self.gf = self.gf[0]

#     @staticmethod
#     def abs_rel_err(a, b):
#         """Return absolute and relative error between a and b.

#         The relative error is a small number when a and b are close, relative
#         to how big they are.

#         Formulas used:
#             abs_err = abs(a - b)
#             rel_err = abs_err / max(abs(a) + abs(b), 1e-8)

#         The denominator is clipped at 1e-8 to avoid dividing by 0 when a and b
#         are both close to 0.

#         The tuple (abs_err, rel_err) is returned
#         """
#         abs_err = abs(a - b)
#         rel_err = abs_err / numpy.maximum(abs(a) + abs(b), 1e-8)
#         # The numpy.asarray are needed as if a or b is a sparse matrix
#         # this would result in a numpy.matrix and not a numpy.ndarray
#         # and the behave differently causing problem later.
#         # In particular a_npy_matrix.flatten().shape == (1, n_element)
#         abs_err = numpy.asarray(abs_err)
#         rel_err = numpy.asarray(rel_err)
#         return (abs_err, rel_err)

#     def abs_rel_errors(self, g_pt):
#         """Return the abs and rel error of gradient estimate `g_pt`

#         `g_pt` must be a list of ndarrays of the same length as self.gf,
#         otherwise a ValueError is raised.

#         Corresponding ndarrays in `g_pt` and `self.gf` must have the same
#         shape or ValueError is raised.

#         """
#         if len(g_pt) != len(self.gf):
#             raise ValueError('argument has wrong number of elements',
#                              len(g_pt))
#         errs = []
#         for i, (a, b) in enumerate(zip(g_pt, self.gf)):
#             if a.shape != b.shape:
#                 raise ValueError('argument element %i has wrong shape %s' % (
#                     i, str((a.shape, b.shape))))
#             errs.append(numeric_grad.abs_rel_err(a, b))
#         return errs

#     def max_err(self, g_pt, abs_tol, rel_tol):
#         """Find the biggest error between g_pt and self.gf.

#         What is measured is the violation of relative and absolute errors,
#         wrt the provided tolerances (abs_tol, rel_tol).
#         A value > 1 means both tolerances are exceeded.

#         Return the argmax of min(abs_err / abs_tol, rel_err / rel_tol) over
#         g_pt, as well as abs_err and rel_err at this point.
#         """
#         pos = []
#         errs = []
#         abs_errs = []
#         rel_errs = []

#         abs_rel_errs = self.abs_rel_errors(g_pt)
#         for abs_err, rel_err in abs_rel_errs:
#             if not numpy.all(numpy.isfinite(abs_err)):
#                 raise ValueError('abs_err not finite', repr(abs_err))
#             if not numpy.all(numpy.isfinite(rel_err)):
#                 raise ValueError('rel_err not finite', repr(rel_err))
#             scaled_err = numpy.minimum(abs_err / abs_tol, rel_err / rel_tol)
#             max_i = scaled_err.argmax()

#             pos.append(max_i)
#             errs.append(scaled_err.flatten()[max_i])
#             abs_errs.append(abs_err.flatten()[max_i])
#             rel_errs.append(rel_err.flatten()[max_i])

#         # max over the arrays in g_pt
#         max_arg = numpy.argmax(errs)
#         max_pos = pos[max_arg]
#         return (max_arg, max_pos, abs_errs[max_arg], rel_errs[max_arg])


# def verify_grad(fun, pt, n_tests=2, rng=None, eps=None,out_type=None, abs_tol=None,rel_tol=None, mode=None, cast_to_output_type=False):
#     # The import is here to prevent circular import.
#     from theano import compile, shared
#     import theano.tensor
#     from theano.tensor import as_tensor_variable, TensorType
#     assert isinstance(pt, (list, tuple))
#     pt = [numpy.array(p) for p in pt]

#     for i, p in enumerate(pt):
#         if p.dtype not in ('float32', 'float64'):
#             raise TypeError(('verify_grad can work only with floating point '
#                 'inputs, but input %i has dtype "%s".') % (i, p.dtype))

#     _type_tol = dict(  # relative error tolerances for different types
#             float32=1e-2,
#             float64=1e-4)

#     if abs_tol is None:
#         abs_tol = __builtin__.max(_type_tol[str(p.dtype)] for p in pt)
#     if rel_tol is None:
#         rel_tol = __builtin__.max(_type_tol[str(p.dtype)] for p in pt)

#     if rng is None:
#         raise TypeError(('rng should be a valid instance of '
#                         'numpy.random.RandomState. You may '
#                          'want to use theano.tests.unittest'
#                          '_tools.verify_grad instead of '
#                          'theano.gradient.verify_grad.'))

#     # We allow input downcast in function, because numeric_grad works in the
#     # most precise dtype used among the inputs, so we may need to cast some.
#     def function(inputs, output):
#         if mode is None:
#             f = compile.function(inputs, output, accept_inplace=True,
#                                  allow_input_downcast=True,
#                                  on_unused_input='ignore')
#         else:
#             f = compile.function(inputs, output, accept_inplace=True,
#                                  allow_input_downcast=True, mode=mode,
#                                  on_unused_input='ignore')
#         return f

#     tensor_pt = [TensorType(
#             as_tensor_variable(p).dtype,
#             as_tensor_variable(p).broadcastable)(name='input %i' % i)
#         for i, p in enumerate(pt)]

#     # fun can be either a function or an actual Op instance
#     o_output = fun(*tensor_pt)

#     if isinstance(o_output, list):
#         raise NotImplementedError(('cant (yet) autotest gradient of fun '
#                                    'with multiple outputs'))
#         # we could make loop over outputs making random projections R for each,
#         # but this doesn't handle the case where not all the outputs are
#         # differentiable... so I leave this as TODO for now -JB.

#     o_fn = function(tensor_pt, o_output)
#     o_fn_out = o_fn(*[p.copy() for p in pt])

#     if isinstance(o_fn_out, tuple) or isinstance(o_fn_out, list):
#         raise TypeError('It seems like you are trying to use verify_grad '
#                 'on an op or a function which outputs a list: there should'
#                 ' be a single (array-like) output instead')

#     # random_projection should not have elements too small,
#     # otherwise too much precision is lost in numerical gradient
#     def random_projection():
#         plain = rng.rand(*o_fn_out.shape) + 0.5
#         if cast_to_output_type:
#             return numpy.array(plain, o_output.dtype)
#         return plain

#     t_r = shared(random_projection())
#     t_r.name = 'random_projection'

#     # random projection of o onto t_r
#     # This sum() is defined above, it's not the builtin sum.
#     cost = theano.tensor.sum(t_r * o_output)

#     cost_fn = function(tensor_pt, cost)

#     symbolic_grad = grad(cost, tensor_pt,
#                          disconnected_inputs='ignore')

#     grad_fn = function(tensor_pt, symbolic_grad)

#     for test_num in xrange(n_tests):
#         try:
#             num_grad = numeric_grad(cost_fn, [p.copy() for p in pt],
#                                     eps, out_type)

#             analytic_grad = grad_fn(*[p.copy() for p in pt])

#             # Since `tensor_pt` is a list, `analytic_grad` should be one too.
#             assert isinstance(analytic_grad, list)

#             max_arg, max_err_pos, max_abs_err, max_rel_err = num_grad.max_err(
#                 analytic_grad, abs_tol, rel_tol)

#             if max_abs_err > abs_tol and max_rel_err > rel_tol:

#                 raise verify_grad.E_grad(max_arg, max_err_pos,
#                                          max_abs_err, max_rel_err,
#                                          abs_tol, rel_tol)

#             # get new random projection for next test
#             if test_num < n_tests - 1:
#                 t_r.set_value(random_projection(), borrow=True)
#         except Exception, e:
#             e.args += ("\nThe error happened with the following inputs:", pt,
#                        "\nThe value of eps is:", eps,
#                        "\nThe out_type is:", out_type)
#             raise


#     #print theano.tests.unittest_tools.verify_grad(T.nnet.GroupDotGrad, [x,w,b,c,gout], n_tests=3)
    
#     return
