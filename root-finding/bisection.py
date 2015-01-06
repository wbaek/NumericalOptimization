#-*- coding: utf-8 -*-
def _verbose(params, scores):
    print 'a=%.5f, b=%.5f, x=%.5f, f(x)=%.5f'%(params[0], params[1], params[2], scores[-1])

def root_finding(function, a, b, epsilon=1e-5, verbose=False):
    ''' root-finding bisectioni method
    >>> root_finding( lambda x: x**3-x-2, 1, 2 )
    1.5213813781738281

    >>> root_finding( lambda x: x**3-x-2, 1, 2, 1e-3, verbose=True )
    a=1.00000, b=2.00000, x=1.50000, f(x)=-0.12500
    a=1.50000, b=2.00000, x=1.75000, f(x)=1.60938
    a=1.50000, b=1.75000, x=1.62500, f(x)=0.66602
    a=1.50000, b=1.62500, x=1.56250, f(x)=0.25220
    a=1.50000, b=1.56250, x=1.53125, f(x)=0.05911
    a=1.50000, b=1.53125, x=1.51562, f(x)=-0.03405
    a=1.51562, b=1.53125, x=1.52344, f(x)=0.01225
    a=1.51562, b=1.52344, x=1.51953, f(x)=-0.01097
    a=1.51953, b=1.52344, x=1.52148, f(x)=0.00062
    1.521484375
    '''
    params = [a, b, (a+b)/2.0]
    scores = [function(param) for param in params]
    if verbose: _verbose(params, scores)
    while (abs(scores[-1]) > epsilon) and (b-a > epsilon):
        if scores[-1] * scores[0] < 0:
            b = params[-1]
        else:
            a = params[-1]
        params = [a, b, (a+b)/2.0]
        scores = [function(param) for param in params]
        if verbose: _verbose(params, scores)
    return params[-1]

if __name__ == '__main__':
    import doctest
    doctest.testmod()
