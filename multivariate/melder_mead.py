#-*- coding: utf-8 -*-
def _verbose(i, params, scores):
    params_str = ', '.join(['(%s)'%(','.join(['%.2f'%p for p in param])) for param in params])
    scores_str = ', '.join(['%.2f'%s for s in scores])
    print 'iter=%03d, params=[%s], scores=[%s]'%(i, params_str, scores_str)

def minimize(function, initial, alpha=1.0, beta=2.0, gamma=0.5, epsilon=1e-5, repeat=int(1e6), verbose=False):
    ''' melder mead method
    >>> minimize( lambda x: (x[0]-0.3)**2 + (x[1]-1.3)**2 + 3.0 ,\
            [(0.0,0.0), (1.0,1.0), (0.0,1.0)] )
    (0.30000068443533018, 1.3000025269138833)

    >>> minimize( lambda x: abs(x[0]-0.3) + abs(x[1]-1.3) + abs(x[2]+0.7) ,\
            [(0.0,0.0,0.0), (1.0,1.0,1.0), (0.0,1.5,0.0), (1.5,2.5,0.0)] )
    (0.30000098578208223, 1.3000002242389832, -0.70000285189454658)

    >>> minimize( lambda x: x[0]**2 - 2*x[0]*x[1] + 2*(x[1]**2) - 6*x[1] + 9 ,\
            [(0.0,0.5), (1.5,1.0), (0.0,1.5)] )
    (3.0000031805013254, 3.0000029243105084)

    >>> minimize( lambda x: (x[0]-0.3)**2 + (x[1]-1.3)**2 + 3.0 ,\
            [(0.3,0.0), (1.0,1.0), (0.0,1.3)], repeat=10, verbose=True )
    iter=000, params=[(0.00,1.30), (1.00,1.00), (0.30,0.00)], scores=[3.09, 3.58, 4.69]
    iter=001, params=[(0.00,1.30), (0.60,1.72), (1.00,1.00)], scores=[3.09, 3.27, 3.58]
    iter=002, params=[(0.00,1.30), (0.65,1.26), (0.60,1.72)], scores=[3.09, 3.12, 3.27]
    iter=003, params=[(0.46,1.50), (0.00,1.30), (0.65,1.26)], scores=[3.07, 3.09, 3.12]
    iter=004, params=[(0.44,1.33), (0.46,1.50), (0.00,1.30)], scores=[3.02, 3.07, 3.09]
    iter=005, params=[(0.23,1.36), (0.44,1.33), (0.46,1.50)], scores=[3.01, 3.02, 3.07]
    iter=006, params=[(0.27,1.26), (0.23,1.36), (0.44,1.33)], scores=[3.00, 3.01, 3.02]
    iter=007, params=[(0.27,1.26), (0.34,1.32), (0.23,1.36)], scores=[3.00, 3.00, 3.01]
    iter=008, params=[(0.27,1.32), (0.27,1.26), (0.34,1.32)], scores=[3.00, 3.00, 3.00]
    iter=009, params=[(0.31,1.31), (0.27,1.32), (0.27,1.26)], scores=[3.00, 3.00, 3.00]
    (0.30558776855468739, 1.3068706512451174)
    '''

    import numpy
    def center_of_params(params):
        return tuple(numpy.average(numpy.array(params[:-1]).transpose(), 1))

    def reflection(params, scores):
        c = center_of_params(params)
        reflection_param = tuple(numpy.array(c) + alpha * numpy.subtract(c, params[-1]))
        return reflection_param, function(reflection_param)

    def expansion(params, scores, reflection_param, reflection_score):
        c = center_of_params(params)
        expansion_param = tuple(numpy.array(c) + beta * numpy.subtract(reflection_param, c))
        return expansion_param, function(expansion_param)

    def contraction(params, scores, reflection_param, reflection_score):
        c = center_of_params(params)
        if reflection_score < scores[-1]: p = reflection_param
        else:                             p = params[-1]
        contraction_param = tuple(numpy.array(c) + gamma * numpy.subtract(p, c))
        contraction_score = function(contraction_param)

        if contraction_score < min(reflection_score, scores[-1]):
            params[-1] = contraction_param
        else:
            params = [tuple(numpy.add(params[0], p)/2.0) for p in params]

        return params, [function(p) for p in params]

    last_index = len(initial)
    scores = [function(p) for p in initial]
    sort_index = [i[0] for i in sorted(enumerate(scores), key=lambda x:x[1])]
    params = [initial[i] for i in sort_index]
    scores = [scores[i] for i in sort_index]
    for i in range(repeat):
        if verbose: _verbose(i, params, scores)
        xr, fr = reflection(params, scores)
        if fr <= scores[0]:
            xe, fe = expansion(params, scores, xr, fr)
            if fe <= fr:
                params[-1] = xe
                scores[-1] = fe
            else:
                params[-1] = xr
                scores[-1] = fr
        elif fr >= scores[-2]:
            params, scores = contraction(params, scores, xr, fr)
        else:
            params[-1] = xr
            scores[-1] = fr

        sort_index = [i[0] for i in sorted(enumerate(scores), key=lambda x:x[1])]
        params = [params[i] for i in sort_index]
        scores = [scores[i] for i in sort_index]

        if epsilon > sum([numpy.linalg.norm(numpy.subtract(params[k], params[k+1]), 2) for k in range(len(params)-1)]):
            break
    return params[0]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
