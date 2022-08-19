import math
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
import scipy.optimize as opt


simSqFun = lambda x: x**2 + 2*x + 1 # 1 -2 +1 = 0 -> min x = -1
diffFun = lambda x, c: 1.0 - math.exp(-(x/c))
diffAbl = lambda x, c: 1.0/c * math.exp(-(x/c))


def simpleOpt():

    xs = np.arange(-3.0, 2.0, 0.1)
    ys = np.array([simSqFun(x) for x in xs])

    optRes = opt.minimize(simSqFun, [100], options={'disp': True})
    print(optRes)

    minPosX = optRes.get("x")[0]
    minPosY = optRes.get("fun")

    optSLSQPRes = opt.minimize(simSqFun, [100], method='SLSQP', options={'disp': True})
    print(optSLSQPRes)

    plt.plot(xs, ys, 'r-', minPosX, minPosY, 'bo')
    plt.show()


def linProg():
    c = [-1, -1]  # maximize x1 + x2 -> minimize -x1 + -x2

    A = [[1, 0], [0, 1]]    # x < 2, y < 3
    b = [2, 3]

    # bounds actually not used here
    x0_bound = (None, 3)
    x1_bound = (None, 3)

    res = opt.linprog(c, A_ub=A, b_ub=b, bounds=[x0_bound, x1_bound])
    print(res)


def difficultEq():
    xs = np.arange(-100, 5.0, 0.0001)
    c = 2
    ys = np.array([diffFun(x, c) for x in xs])
    ds = np.array([diffAbl(x, c) for x in xs])

    plt.plot(xs, ys, 'r-',   xs, ds, 'b-')
    plt.show()

if __name__ == "__main__":
    difficultEq()