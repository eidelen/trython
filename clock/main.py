# This is a sample Python script.
import math
import matplotlib
## Agg backend runs without a display and is thus suitable for use on backend servers
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

sv = math.pi / 30.0
mv = math.pi / 1800.0
hv = math.pi / (6*3600.0)

def compute_hour_cycles():
    for hour in range(0, 12):
        t_meet = hour * 3600 / 11.0
        ang_meet = t_meet * mv
        ang_second_arm = (t_meet * sv) % (2 * math.pi)
        print(ang_meet, ang_second_arm, ang_meet - ang_second_arm)

def generate_plots():
    t = np.linspace(0, 1*3600, 100000)

    # Define your functions
    s = np.mod( t * sv, 2*math.pi )
    m = np.mod( t * mv, 2*math.pi )
    h = np.mod( t * hv, 2*math.pi )

    # Make the plots for each function
    plt.plot(t, s, color='blue', label='sec')
    plt.plot(t, m, color='red', label='min')
    plt.plot(t, h, color='green', label='hour')

    # Add labels and show the plot
    plt.xlabel('Time')
    plt.ylabel('Angle')
    plt.legend()
    plt.show()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    compute_hour_cycles()
    generate_plots()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
