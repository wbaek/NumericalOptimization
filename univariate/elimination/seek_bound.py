#-*- coding: utf-8 -*-
def _verbose(params, scores):
    print 'params=' + ','.join(['%.2f'%p for p in params]) + ', scores=' + ','.join(['%.2f'%s for s in scores])

import random
def seek_bound(function, x=random.random(), d=random.random(), verbose=False):
    ''' seek initial bound
    >>> seek_bound( lambda x: abs(x-0.3), 0.35, 0.2, verbose=True )
    params=0.15,0.35,0.55, scores=0.15,0.05,0.25
    (0.14999999999999997, 0.55)

    >>> seek_bound( lambda x: abs(x-0.3), 0.1, 0.3, verbose=True )
    params=-0.20,0.10,0.40, scores=0.50,0.20,0.10
    params=0.10,0.40,1.00, scores=0.20,0.10,0.70
    (0.1, 1.0)

    >>> seek_bound( lambda x: abs(x-0.3), 0.5, 0.3, verbose=True )
    params=0.20,0.50,0.80, scores=0.10,0.20,0.50
    params=0.50,0.20,-0.40, scores=0.20,0.10,0.70
    (-0.39999999999999997, 0.5)

    >>> seek_bound( lambda x: abs(x-0.3), 2.0, 0.3, verbose=True )
    params=1.70,2.00,2.30, scores=1.40,1.70,2.00
    params=2.00,1.70,1.10, scores=1.70,1.40,0.80
    params=1.70,1.10,-0.10, scores=1.40,0.80,0.40
    params=1.10,-0.10,-2.50, scores=0.80,0.40,2.80
    (-2.5, 1.2)
    '''
    params = [x-d, x, x+d]
    scores = [function(p) for p in params]
    if verbose: _verbose(params, scores)

    if scores[0] >= scores[1] and scores[1] <= scores[2]:
        pass
    else:
        if scores[0] <= scores[2]:
            d = -d
            params.reverse()

        k = 0
        while True:
            k += 1
            params = [params[1], params[2], params[2] + (2 ** k) * d]
            scores = [function(p) for p in params]
            if verbose: _verbose(params, scores)
            if scores[2] >= scores[1]:
                if d < 0: params.reverse()
                break;

    return (params[0], params[-1])

if __name__ == '__main__':
    import doctest
    doctest.testmod()
