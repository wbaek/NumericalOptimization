#-*- coding: utf-8 -*-
def _verbose(i, param, score):
    params_str = '(%s)'%(','.join(['%.2f'%p for p in param]))
    scores_str = '%.2f'%score
    print 'iter=%03d, params=%s, scores=%s'%(i, params_str, scores_str)

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from univariate.elimination import seek_bound
from univariate.elimination import golden_section

def minimize(function, initial, epsilon=1e-6, repeat=int(1e4), verbose=False):
    ''' powell's method
    >>> ['%.2f'%v for v in minimize( lambda x: (x[0]-0.3)**2 + (x[1]-1.3)**2 + 3.0, (0.0,0.0) )]
    ['0.30', '1.30']

    >>> ['%.2f'%v for v in minimize( lambda x: abs(x[0]-0.3) + abs(x[1]-1.3) + abs(x[2]+0.7), (0.0,0.0,0.0) )]
    ['0.30', '1.30', '-0.70']

    >>> ['%.2f'%v for v in minimize( lambda x: x[0]**2 - 2*x[0]*x[1] + 2*(x[1]**2) - 6*x[1] + 9, (0.0,0.5) )]
    ['3.00', '3.00']

    >>> ['%.2f'%v for v in minimize( lambda x: (x[0]-0.3)**2 + (x[1]-1.3)**2 + 3.0, (0.0,0.0), verbose=True )]
    iter=000, params=(0.30,0.00), scores=4.69
    iter=001, params=(0.30,1.30), scores=3.00
    iter=002, params=(0.30,1.30), scores=3.00
    ['0.30', '1.30']
    '''

    import numpy
    dimension = len(initial)
    unit_vector_list = [ numpy.array([1.0 if i == j else 0.0 for j in range(dimension)])
            for i in range(dimension) ]
    point_list = [numpy.array( initial ) for i in range(dimension+1)]

    minimized_point = numpy.array( initial )
    for i in range(repeat):
        for k in range(dimension):
            unimodal_function = lambda gamma: function(point_list[k] + gamma * unit_vector_list[k])
            bound = seek_bound.seek_bound(unimodal_function)
            minimized_gamma = golden_section.minimize(unimodal_function, bound[0], bound[1])
            point_list[k+1] = point_list[k] + minimized_gamma * unit_vector_list[k]

        unit_vector_list = [unit_vector_list[k+1] for k in range(dimension-1)]
        unit_vector_list.append( point_list[k+1] - point_list[0] )

        unimodal_function = lambda gamma: function(point_list[0] + gamma * unit_vector_list[dimension-1])
        bound = seek_bound.seek_bound(unimodal_function)
        minimized_gamma = golden_section.minimize(unimodal_function, bound[0], bound[1])

        _ = minimized_point
        minimized_point = point_list[0] + minimized_gamma * unit_vector_list[dimension-1]
        point_list[0] = minimized_point
        if verbose: _verbose(i, minimized_point, function(minimized_point))
        if numpy.linalg.norm(numpy.subtract(_, minimized_point), 2) < epsilon:
            break

    return tuple(minimized_point)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
