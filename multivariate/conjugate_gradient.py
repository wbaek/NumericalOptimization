#-*- coding: utf-8 -*-
def _verbose(i, param, direction, step_length, score):
    params_str = '(%s)'%(','.join(['%.2f'%p for p in param]))
    direction_str = '(%s)'%(','.join(['%.2f'%d for d in direction]))
    scores_str = '%.2f'%score
    print 'iter=%03d, params=%s, direction=%s, step_length=%.5f, scores=%s'%(
        i, params_str, direction_str, step_length, scores_str)

import steepest_descent
import numpy
def minimize(function, derivate, initial, epsilon=1e-6, repeat=int(1e3), verbose=False):
    ''' steepest descent
    >>> ['%.2f'%v for v in minimize( \
            lambda x: x[0]**2 - 2*x[0]*x[1] + 2*(x[1]**2) - 6*x[1] + 9, \
            lambda x: (2*x[0] - 2*x[1], 4*x[1] - 2*x[0] - 6), \
            (0.0,5.5) )]
    ['3.00', '3.00']

    >>> ['%.2f'%v for v in minimize( \
            lambda x: (x[0]-0.3)**2 + (x[1]-1.3)**2 + (x[2]+0.7)**2, \
            lambda x: (2*x[0]-0.6, 2*x[1]-2.6, 2*x[2]+1.4), \
            (0.0, 0.0, 0.0) )]
    ['0.30', '1.30', '-0.70']

    >>> ['%.2f'%v for v in minimize( \
            lambda x: (x[0]-0.3)**2 + (x[1]-1.3)**2 + 3.0, \
            lambda x: (2*x[0] - 0.6, 2*x[1] -2.6), \
            (0.0, 0.0), repeat=10, verbose=True)]
    iter=000, params=(0.00,0.00), direction=(0.22,0.97), step_length=0.53367, scores=4.78
    iter=001, params=(0.12,0.52), direction=(0.22,0.97), step_length=0.39220, scores=3.64
    iter=002, params=(0.21,0.90), direction=(0.22,0.97), step_length=0.21535, scores=3.17
    iter=003, params=(0.26,1.11), direction=(0.22,0.97), step_length=0.12184, scores=3.04
    iter=004, params=(0.28,1.23), direction=(0.22,0.97), step_length=0.05560, scores=3.01
    iter=005, params=(0.30,1.28), direction=(0.22,0.97), step_length=0.01005, scores=3.00
    iter=006, params=(0.30,1.29), direction=(0.22,0.97), step_length=0.00361, scores=3.00
    iter=007, params=(0.30,1.30), direction=(0.22,0.97), step_length=0.00129, scores=3.00
    iter=008, params=(0.30,1.30), direction=(0.22,0.97), step_length=0.00040, scores=3.00
    iter=009, params=(0.30,1.30), direction=(0.22,0.97), step_length=0.00010, scores=3.00
    ['0.30', '1.30']
    '''

    pt = numpy.array(initial)
    gradient = [numpy.array(derivate(pt))]
    direction = -gradient[0]

    for i in range(repeat):
        step_length = steepest_descent.find_step_length(function, derivate, pt, direction,
                check=steepest_descent.strong_wolfe_condition)
        if verbose: _verbose(i, pt, direction, step_length, function(pt))

        pt = pt + step_length * direction
        gradient.append( numpy.array(derivate(pt)) )
        gradient = gradient[-2:]

        beta = gradient[1].dot( gradient[1] ) / float(gradient[0].dot( gradient[0] ))
        direction = (-gradient[1]) + (beta * direction)

        length_of_gradient = numpy.linalg.norm( gradient[1], 2 )
        if step_length < epsilon or length_of_gradient < epsilon:
            break

    return tuple(pt)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
