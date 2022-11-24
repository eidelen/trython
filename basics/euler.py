import math

steps = 100000
accum = 1.0
q = 1.0

for n in range(1, steps):

    accum += q / steps * accum

    print(accum, math.fabs(accum - math.e))

print(1 * (q/steps)**steps)