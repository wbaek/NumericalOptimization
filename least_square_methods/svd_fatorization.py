#-*- coding: utf-8 -*-
def _verbose(i, param, direction, hessian, step_length, score):
    params_str = '(%s)'%(','.join(['%.2f'%p for p in param]))
    direction_str = '(%s)'%(','.join(['%.2f'%d for d in direction]))
    hessian_str = '(%s)'%(','.join(
        ['[' + ','.join(['%.2f'%h for h in hessian_row]) + ']' for hessian_row in hessian]) )
    scores_str = '%.2f'%score
    print 'iter=%03d, params=%s, direction=%s, hessian_inv=%s, step_length=%.5f, scores=%s'%(
            i, params_str, direction_str, hessian_str, step_length, scores_str)

import numpy
def minimize(jacobian, input_list, output_list):
    ''' svd factorization
    >>> def f(model, _in, _out):
    ...    return sum([1.0/2.0 * (model[0]*x[0]+model[1] - _out[i])**2 for i, x in enumerate(_in)])
    >>> param = minimize( \
            lambda x: (x[0], 1.0), \
            [[0.0], [1.0], [2.0]], [1.0, 1.5, 2.0])
    >>> ['%.2f'%v for v in param]
    ['0.50', '1.00']
    >>> '%.2f'%f(param, [[0.0], [1.0], [2.0]], [1.0, 1.5, 2.0])
    '0.00'

    >>> def f(model, _in, _out):
    ...     return sum([1.0/2.0 * (model[0]*x[0]+model[1] - _out[i])**2 for i, x in enumerate(_in)])
    >>> param = minimize( \
            lambda x: (x[0], 1.0), \
            [[0.0], [1.0], [2.0], [3.0]], [1.0, 2.0, 1.5, 2.5])
    >>> ['%.2f'%v for v in param]
    ['0.40', '1.15']
    >>> '%.2f'%f(param, [[0.0], [1.0], [2.0], [3.0]], [1.0, 2.0, 1.5, 2.5])
    '0.22'

    function(t, x, y) = 1.0/2.0 * (t[0]*x[0]+t[1]*x[1]+t[2] - y)**2
    >>> ['%.2f'%abs(v) for v in minimize( \
            lambda x: (x[0], x[1], 1.0), \
            [[0.0, 4.0], [1.0, 2.0], [2.0, 1.0], [3.0, 2.0]], [1.0, 1.5, 2.0, 2.5])]
    ['0.50', '0.00', '1.00']
    '''

    J = numpy.mat([jacobian(x) for x in input_list])
    U,s,V = numpy.linalg.svd(J.transpose() * J)
    S = numpy.diag(s)

    model = V.transpose() * numpy.linalg.inv(S) * U.transpose() * (J.transpose() * numpy.mat(output_list).transpose())
    return tuple(model[:J.shape[0]])

if __name__ == '__main__':
    import doctest
    doctest.testmod()
