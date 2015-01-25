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
def minimize(function, derivate, derivate_2nd, initial, epsilon=1e-6, repeat=int(1e4), verbose=False):
    ''' steepest descent
    >>> ['%.2f'%v for v in minimize( \
            lambda x: x[0]**2 - 2*x[0]*x[1] + 2*(x[1]**2) - 6*x[1] + 9, \
            lambda x: (2*x[0] - 2*x[1], 4*x[1] - 2*x[0] - 6), \
            lambda x: ((2, -2), (-2, 4)), \
            (0.0,5.5) )]
    ['3.00', '3.00']

    >>> ['%.2f'%v for v in minimize( \
            lambda x: (x[0]-0.3)**2 + (x[1]-1.3)**2 + (x[2]+0.7)**2, \
            lambda x: (2*x[0]-0.6, 2*x[1]-2.6, 2*x[2]+1.4), \
            lambda x: ((2, 0, 0), (0, 2, 0), (0, 0, 2)), \
            (0.0, 0.0, 0.0) )]
    ['0.30', '1.30', '-0.70']

    >>> ['%.2f'%v for v in minimize( \
            lambda x: (x[0]-0.3)**2 + (x[1]-1.3)**2 + 3.0, \
            lambda x: (2*x[0] - 0.6, 2*x[1] -2.6), \
            lambda x: ((2, 0), (0, 2)), \
            (0.0, 0.0), repeat=10, verbose=True)]
    iter=000, params=(0.00,0.00), direction=(0.30,1.30), hessian_inv=([0.50,0.00],[0.00,0.50]), step_length=1.00000, scores=4.78
    iter=001, params=(0.30,1.30), direction=(-0.00,-0.00), hessian_inv=([0.50,0.00],[0.00,0.50]), step_length=1.00000, scores=3.00
    ['0.30', '1.30']
    '''

    pt = numpy.array( initial )
    step_length = 1.0

    for i in range(repeat):
        gradient = numpy.array(derivate( pt ))
        hessian = numpy.array(derivate_2nd( pt ))
        inverted_hessian = numpy.linalg.inv( hessian )
        direction = - numpy.dot( inverted_hessian, gradient)

        if verbose: _verbose(i, pt, direction, inverted_hessian, step_length, function(pt))

        length_of_gradient = numpy.linalg.norm( gradient, 2 )
        if step_length < epsilon or length_of_gradient < epsilon:
            break

        pt = pt + direction * step_length

    return tuple(pt)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
