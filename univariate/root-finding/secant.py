#-*- coding: utf-8 -*-
def _verbose(params, score):
    print 'a=%.5f, b=%.5f, f(x)=%.5f'%(params[0], params[1], score)

def approximate_derivate(function, a, b):
    return (function(a) - function(b)) / (a - b)

def root_finding(function, a, b, epsilon=1e-5, verbose=False):
    ''' root-finding bisection method
    >>> root_finding( lambda x: x**3-x-2, 1.0, 2.0 )
    1.5213796791513041

    >>> root_finding( lambda x: x**3-x-2, 1.0, 2.0, 1e-5, verbose=True )
    a=1.00000, b=2.00000, f(x)=-2.00000
    a=1.33333, b=1.00000, f(x)=-0.96296
    a=1.64286, b=1.33333, f(x)=0.79118
    a=1.50325, b=1.64286, f(x)=-0.10626
    a=1.51978, b=1.50325, f(x)=-0.00949
    a=1.52140, b=1.51978, f(x)=0.00013
    a=1.52138, b=1.52140, f(x)=-0.00000
    1.5213796791513041
    '''

    params = [a, b]
    score, delta = 1, 1
    while (abs(score) > epsilon) and (abs(delta) > epsilon):
        score = function(params[0])
        delta = score / approximate_derivate(function, params[0], params[1])
        if verbose: _verbose(params, score)
        params[1] = params[0]
        params[0] = params[0] - delta
    return params[-1]

if __name__ == '__main__':
    import doctest
    doctest.testmod()
