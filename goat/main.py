from math import asin, sin, cos, pi, sqrt
from scipy.optimize import root_scalar


def g(a):
    val = pi/2.0 - sin(a) - pi*cos(a) + a * cos(a)
    return val

def f(r):
    R = 1.0
    a = 2 * asin(r / (2 * R))
    b = (pi - a) / 2
    A1 = r ** 2 * (b - cos(b) * sin(b))
    A2 = R ** 2 * (a - cos(a) * sin(a))
    AT = (R ** 2 * pi) / 2
    residual = A1 + A2 - AT
    return residual

result = root_scalar(f, bracket=[0.5, 2], method='brentq')
print("Res1", result.root)

a = root_scalar(g, bracket=[0.5, 1.5], method='brentq').root
print("Res2", a, "r:", 1 * sqrt( 2 * (1 - cos(a)) ))