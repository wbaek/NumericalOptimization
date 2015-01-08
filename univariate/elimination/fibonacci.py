#-*- coding: utf-8 -*-
def _verbose(a, b, x1, x2, f1, f2):
    print 'a=%.5f, b=%.5f, x1=%.5f, f(x1)=%.5f, x2=%.5f, f(x2)=%.5f'%(a, b, x1, f1, x2, f2)

def minimize(function, a, b, N, epsilon=1e-5, verbose=False):
    ''' fibonacci elimination
    >>> minimize( lambda x: abs(x-0.3), 0.0, 1.0, 20 )
    0.3000182715147086

    >>> minimize( lambda x: abs(x-0.3), 0.0, 1.0, 10, verbose=True )
    a=0.00000, b=1.00000, x1=0.38202, f(x1)=0.08202, x2=0.61798, f(x2)=0.31798
    a=0.00000, b=0.61798, x1=0.23596, f(x1)=0.06404, x2=0.38202, f(x2)=0.08202
    a=0.00000, b=0.38202, x1=0.14607, f(x1)=0.15393, x2=0.23596, f(x2)=0.06404
    a=0.14607, b=0.38202, x1=0.23596, f(x1)=0.06404, x2=0.29213, f(x2)=0.00787
    a=0.23596, b=0.38202, x1=0.29213, f(x1)=0.00787, x2=0.32584, f(x2)=0.02584
    a=0.23596, b=0.32584, x1=0.26966, f(x1)=0.03034, x2=0.29213, f(x2)=0.00787
    a=0.26966, b=0.32584, x1=0.29213, f(x1)=0.00787, x2=0.30337, f(x2)=0.00337
    a=0.29213, b=0.32584, x1=0.30337, f(x1)=0.00337, x2=0.31461, f(x2)=0.01461
    a=0.29213, b=0.31461, x1=0.30337, f(x1)=0.00337, x2=0.30337, f(x2)=0.00338
    0.3033707865168539
    '''

    fibonacci_numubers = [1.0 for i in range(N+1)]
    for i in range(N+1):
        fibonacci_numubers[i] = (fibonacci_numubers[i-1] + fibonacci_numubers[i-2]) if i >= 2 else 1.0

    L = b-a
    for i in range(N, 1, -1):
        L = b-a
        x1 = a + (fibonacci_numubers[i-2] / fibonacci_numubers[i] * L)
        x2 = b - (fibonacci_numubers[i-2] / fibonacci_numubers[i] * L)
        f1 = function(x1)
        f2 = function(x2) + (epsilon if i == 2 else 0.0)
        if verbose: _verbose(a, b, x1, x2, f1, f2)
        if f1 > f2:
            a = x1
        else:
            b = x2
    return x1

if __name__ == '__main__':
    import doctest
    doctest.testmod()
