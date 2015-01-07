#-*- coding: utf-8 -*-
def _verbose(param, score):
    print 'x=%.5f, f(x)=%.5f'%(param, score)

def root_finding(function, derivate, x, epsilon=1e-5, verbose=False):
    ''' root-finding bisectioni method
    >>> root_finding( lambda x: x**3-x-2, lambda x: 3*(x**2)-1, 1.0 )
    1.521379709733148

    >>> root_finding( lambda x: x**3-x-2, lambda x: 3*(x**2)-1, 1.0, epsilon=1e-5, verbose=True )
    x=2.00000, f(x)=4.00000
    x=1.63636, f(x)=0.74530
    x=1.53039, f(x)=0.05394
    x=1.52144, f(x)=0.00037
    x=1.52138, f(x)=0.00000
    1.521379709733148
    '''

    score, delta = 1, 1
    while (abs(score) > epsilon) and (abs(delta) > epsilon):
        delta = function(x) / derivate(x)
        x = x - delta
        score = function(x)
        if verbose: _verbose(x, score)
    return x


if __name__ == '__main__':
    import doctest
    doctest.testmod()
