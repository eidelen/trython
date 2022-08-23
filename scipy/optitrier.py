import math
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
import scipy.optimize as opt


simSqFun = lambda x: x**2 + 2*x + 1 # 1 -2 +1 = 0 -> min x = -1

diffFun = lambda x, c: math.exp(-(x/c))
diffAbl = lambda x, c: -1.0/c * math.exp(-(x/c))


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
    xs = np.arange(0, 3.0, 0.0001)
    c = 1
    ys = np.array([diffFun(x, c) for x in xs])
    ds = np.array([diffAbl(x, c) for x in xs])

    specific = lambda x: diffFun(x, 1)
    specificDiff = lambda x: diffAbl(x, 1)

    optSLSQPRes = opt.minimize(specific, [1], method='SLSQP', jac=specificDiff, options={'disp': True}, bounds=[(0, 5)])
    print(optSLSQPRes)

    plt.plot(xs, ys, 'r-',   xs, ds, 'b-')
    plt.show()


def optimizeWithContrainsts():

    # minimize f(x) = exp(-x/1) + exp(-y/2), where x + y = 2.0
    tot = 2.0
    s1 = lambda x: diffFun(x, 1)
    s2 = lambda y: diffFun(y, 2)
    obj = lambda x: s1(x[0]) + s2(x[1])
    objAbl = lambda x: [diffAbl(x[0], 1), diffAbl(x[1], 2)]

    xs = np.arange(0, tot, 0.0001)
    objVal = np.array([obj([x, tot - x]) for x in xs])
    minPos = np.argmin(objVal)
    xMin = xs[minPos]
    print("Min at pos = ", xMin, tot - xMin)

    cons = ({'type': 'eq', 'fun': lambda x: tot - sum(x)})

    # optimizer
    optSLSQPRes = opt.minimize(obj, [7, 7], method='SLSQP', options={'disp': True}, jac=objAbl, constraints=cons, bounds=[(0, tot), (0, tot)])
    print(optSLSQPRes)

    plt.plot(xs, objVal, 'r-')
    plt.show()


def meanTimeOfDetection(sweepWidth: float, velocity: float, area: float):
    return area / (velocity * sweepWidth) # denoted T in formula

def minFuncProb(x, w, v, a, p):
    return p * diffFun(x, meanTimeOfDetection(w, v, a))


def objectiveFunctionMin(ts, areas, w, v):
    sum = 0
    for t, ar in zip(ts, areas):
        _, _, asize, prob = ar
        sum += minFuncProb(t, w, v, asize, prob)
    return sum

def deriveObjectiveFunction(ts, areas, w, v):
    drv = []
    for t, ar in zip(ts, areas):
        _, _, asize, prob = ar
        d = prob * diffAbl(t, meanTimeOfDetection(w, v, asize))
        drv.append(d)
    return drv

def optLargeProblemConst():
    w = 200  # m
    v = 10  # m/s
    t = 6 * 60 * 60  # s

    areas = [("A1", "urban", 5000000, 0.45), ("A2", "mountain", 10000000, 0.1), ("A3", "water", 5000000, 0.45 )]
    nArea = len(areas)

    optFunc = lambda x: objectiveFunctionMin(x, areas, w, v)
    drvFunc = lambda x: deriveObjectiveFunction(x, areas, w, v)

    cons = ({'type': 'eq', 'fun': lambda x: t - sum(x)})

    # optimizer
    optSLSQPRes = opt.minimize(optFunc, [x for x in range(nArea)], method='SLSQP', options={'ftol': 1e-20, 'disp': True},
                               jac=drvFunc, constraints=cons, bounds=[(0, t) for x in range(nArea)])
    print(optSLSQPRes)



if __name__ == "__main__":
    optLargeProblemConst()