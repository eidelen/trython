from math import asin, sin, cos, pi
from scipy.optimize import root_scalar

def f(r):
    R = 1.0
    a = 2 * asin(r / (2 * R))
    b = (pi - a) / 2
    A1 = r ** 2 * (b - cos(b) * sin(b))
    A2 = R ** 2 * (a - cos(a) * sin(a))
    AT = (R ** 2 * pi) / 2
    residual = A1 + A2 - AT
    print(AT, A1 + A2, residual)
    return residual

result = root_scalar(f, bracket=[0.5, 2], method='brentq')
print(result.root)