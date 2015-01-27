#-*- coding: utf-8 -*-
def _verbose(i, param, direction, step_length, score):
    params_str = '(%s)'%(','.join(['%.2f'%p for p in param]))
    direction_str = '(%s)'%(','.join(['%.2f'%d for d in direction]))
    scores_str = '%.2f'%score
    print 'iter=%03d, params=%s, direction=%s, step_length=%.5f, scores=%s'%(i, params_str, direction_str, step_length, scores_str)

import numpy
def wolfe_condition(score_pt, score_alpha_pt, gradient_pt, gradient_alpha_pt, alpha, direction,
        constant1=0.6, constant2=0.9):
    if score_alpha_pt - score_pt <= constant1 * alpha * numpy.dot(gradient_pt, direction) \
    and numpy.dot(gradient_alpha_pt, direction) >= constant2 * numpy.dot(gradient_pt, direction):
        return True
    else:
        return False

def strong_wolfe_condition(score_pt, score_alpha_pt, gradient_pt, gradient_alpha_pt, alpha, direction,
        constant1=0.6, constant2=0.9):
    if score_alpha_pt - score_pt <= constant1 * alpha * numpy.dot(gradient_pt, direction) \
    and abs(numpy.dot(gradient_alpha_pt, direction)) <= - constant2 * numpy.dot(gradient_pt, direction):
        return True
    else:
        return False

def find_step_length(function, derivate, pt, direction,
        tau=0.8, repeat=int(1e2), initial_length=1.0, check=wolfe_condition):
    score_at_pt = function(pt)
    gradient_at_pt = derivate(pt)

    length_of_direction = numpy.linalg.norm(direction, 2)
    direction /= length_of_direction

    alpha = 1.0
    start_pt = pt
    last_pt = pt + direction * initial_length * length_of_direction
    for i in range(repeat):
        length = numpy.linalg.norm( last_pt - start_pt, 2 )
        pt1 = start_pt + direction * ( (1.0 - tau) * length )
        pt2 = start_pt + direction * ( (tau) * length )
        score_pt1 = function( pt1 )
        score_pt2 = function( pt2 )

        # wolfe condition
        _ = (1.0 - tau) * length
        if check(score_at_pt, score_pt1, gradient_at_pt, derivate(pt1), _, direction):
            alpha = _
            break
        _ = (tau) * length
        if check(score_at_pt, score_pt2, gradient_at_pt, derivate(pt2), _, direction):
            alpha = _
            break

        if score_pt1 >= score_pt2: start_pt = pt1
        else: last_pt = pt2

    return alpha

def minimize(function, derivate, initial, epsilon=1e-6, repeat=int(1e4), verbose=False):
    ''' steepest descent
    >>> ['%.2f'%v for v in minimize( \
            lambda x: x[0]**2 - 2*x[0]*x[1] + 2*(x[1]**2) - 6*x[1] + 9, \
            lambda x: (2*x[0] - 2*x[1], 4*x[1] - 2*x[0] - 6), \
            (0.0,5.5) )]
    ['3.00', '3.00']

    >>> ['%.2f'%v for v in minimize( \
            lambda x: (x[0]-0.3)**2 + (x[1]-1.3)**2 + (x[2]+0.7)**2, \
            lambda x: (2*x[0]-0.6, 2*x[1]-2.6, 2*x[2]+1.4), \
            (0.0, 0.0, 0.0))]
    ['0.30', '1.30', '-0.70']

    >>> ['%.2f'%v for v in minimize( \
            lambda x: (x[0]-0.3)**2 + (x[1]-1.3)**2 + 3.0, \
            lambda x: (2*x[0] - 0.6, 2*x[1] -2.6), \
            (0.0, 0.0), repeat=10, verbose=True)]
    iter=000, params=(0.00,0.00), direction=(0.22,0.97), step_length=0.80000, scores=4.78
    iter=001, params=(0.18,0.78), direction=(0.22,0.97), step_length=0.20000, scores=3.29
    iter=002, params=(0.22,0.97), direction=(0.22,0.97), step_length=0.20000, scores=3.11
    iter=003, params=(0.27,1.17), direction=(0.22,0.97), step_length=0.06554, scores=3.02
    iter=004, params=(0.28,1.23), direction=(0.22,0.97), step_length=0.03355, scores=3.00
    iter=005, params=(0.29,1.27), direction=(0.22,0.97), step_length=0.01718, scores=3.00
    iter=006, params=(0.30,1.28), direction=(0.22,0.97), step_length=0.00880, scores=3.00
    iter=007, params=(0.30,1.29), direction=(0.22,0.97), step_length=0.00450, scores=3.00
    iter=008, params=(0.30,1.30), direction=(0.22,0.97), step_length=0.00231, scores=3.00
    iter=009, params=(0.30,1.30), direction=(0.22,0.97), step_length=0.00118, scores=3.00
    ['0.30', '1.30']
    '''

    pt = numpy.array( initial )
    for i in range(repeat):
        descending_direction = - numpy.array(derivate( pt ))
        length_of_gradient = numpy.linalg.norm(descending_direction, 2)
        descending_direction /= length_of_gradient

        step_length = find_step_length(function, derivate, pt, descending_direction)
        if verbose: _verbose(i, pt, descending_direction, step_length, function(pt))

        pt = pt + descending_direction * step_length
        if step_length < epsilon or length_of_gradient < epsilon:
            break

    return tuple(pt)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
