import matplotlib.pyplot as plt
import math
import datetime

h0 = 500
g = 9.81

h_before = h0
start_time = datetime.datetime.now()

while True:

    # how much time passed
    t = (datetime.datetime.now() - start_time).total_seconds()

    # Torricelli's law
    h = 0.5*g*(t**2) - math.sqrt(2*g*h0)*t + h0

    # stop simulation when height starts rising (quadratic equation)
    if h > h_before:
        break
    h_before = h

    # update chart -> insanely slow
    plt.clf()
    plt.axis([0.6, 1, 0, h0])
    plt.bar([1], [h], align='center', alpha=0.5)
    plt.pause(0.01)
