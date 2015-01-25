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
def SR1(derivate, point, direction, step_length, Bk):
    sk = direction * step_length
    yk = numpy.array(derivate(point + sk)) - numpy.array(derivate(point))
    tk = sk - Bk.dot( yk )

    # check positive definite
    if abs(numpy.dot(tk, yk)) > 0:
        _ = numpy.matrix( tk )
        _ = _.transpose().dot( _ ) / _.dot( yk )
        Bk += _
    return Bk

def BFGS(derivate, point, direction, step_length, Bk):
    sk = numpy.array( direction * step_length )
    yk = numpy.array(derivate(point + sk)) - numpy.array(derivate(point))
    rho = 1.0 / sk.dot( yk.transpose() )

    identity = numpy.identity( Bk.shape[0] )
    sk_m = numpy.matrix(sk)
    yk_m = numpy.matrix(yk)
    _ = (identity - rho * sk_m.transpose() * yk_m) * Bk * (identity - rho * yk_m.transpose() * sk_m)
    _ += (rho * sk_m.transpose() * sk_m)

    return numpy.array(_)


def minimize(function, derivate, initial, update_method=SR1, epsilon=1e-6, repeat=int(1e1), verbose=False):
    ''' quasi-newton's method
    >>> [['%.2f'%v for v in minimize( \
            lambda x: (x[0]-0.3)**2 + (x[1]-1.3)**2 + 3.0, \
            lambda x: (2*x[0] - 0.6, 2*x[1] -2.6), \
            (0.0, 0.0), method )] for method in [SR1, BFGS]]
    [['0.30', '1.30'], ['0.30', '1.30']]

    >>> [['%.2f'%v for v in minimize( \
            lambda x: (x[0]-0.3)**2 + (x[1]-1.3)**2 + (x[2]+0.7)**2, \
            lambda x: (2*x[0]-0.6, 2*x[1]-2.6, 2*x[2]+1.4), \
            (0.0, 0.0, 0.0), method )] for method in [SR1, BFGS]]
    [['0.30', '1.30', '-0.70'], ['0.30', '1.30', '-0.70']]

    >>> ['%.2f'%v for v in minimize( \
            lambda x: x[0]**2 - 2*x[0]*x[1] + 2*(x[1]**2) - 6*x[1] + 9, \
            lambda x: (2*x[0] - 2*x[1], 4*x[1] - 2*x[0] - 6), \
            (0.0, 5.5), update_method=SR1, repeat=10, verbose=True)]
    iter=000, params=(0.00,5.50), direction=(11.00,-16.00), hessian_inv=([0.78,0.36],[0.36,0.41]), step_length=1.00000, scores=36.50
    iter=001, params=(11.00,-10.50), direction=(-8.21,13.37), hessian_inv=([1.00,0.50],[0.50,0.50]), step_length=1.00000, scores=644.50
    iter=002, params=(2.79,2.87), direction=(0.21,0.13), hessian_inv=([1.00,0.50],[0.50,0.50]), step_length=1.00000, scores=0.02
    iter=003, params=(3.00,3.00), direction=(-0.00,-0.00), hessian_inv=([1.00,0.50],[0.50,0.50]), step_length=1.00000, scores=0.00
    ['3.00', '3.00']

    >>> ['%.2f'%v for v in minimize( \
            lambda x: x[0]**2 - 2*x[0]*x[1] + 2*(x[1]**2) - 6*x[1] + 9, \
            lambda x: (2*x[0] - 2*x[1], 4*x[1] - 2*x[0] - 6), \
            (0.0, 5.5), update_method=BFGS, repeat=10, verbose=True)]
    iter=000, params=(0.00,5.50), direction=(11.00,-16.00), hessian_inv=([0.78,0.36],[0.36,0.41]), step_length=1.00000, scores=36.50
    iter=001, params=(11.00,-10.50), direction=(-8.21,13.37), hessian_inv=([0.78,0.36],[0.36,0.42]), step_length=1.00000, scores=644.50
    iter=002, params=(2.79,2.87), direction=(0.16,0.10), hessian_inv=([1.00,0.50],[0.50,0.50]), step_length=1.00000, scores=0.02
    iter=003, params=(2.95,2.97), direction=(0.05,0.03), hessian_inv=([1.00,0.50],[0.50,0.50]), step_length=1.00000, scores=0.00
    iter=004, params=(3.00,3.00), direction=(0.00,-0.00), hessian_inv=([1.00,0.50],[0.50,0.50]), step_length=1.00000, scores=0.00
    ['3.00', '3.00']
    '''

    pt = numpy.array( initial )
    dimension = len(initial)
    direction = numpy.zeros( dimension )
    step_length = 1.0
    inverted_approximated_hessian = numpy.identity( dimension )

    for i in range(repeat):
        gradient = numpy.array(derivate( pt ))
        direction = - numpy.dot(inverted_approximated_hessian, gradient)

        inverted_approximated_hessian = update_method(
                    derivate, pt, direction, step_length, inverted_approximated_hessian)

        if verbose: _verbose(i, pt, direction, inverted_approximated_hessian, step_length, function(pt))
        pt = pt + direction * step_length

        length_of_gradient = numpy.linalg.norm( gradient, 2 )
        if step_length < epsilon or length_of_gradient < epsilon:
            break

    return tuple(pt)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
