import numpy as np
import matplotlib.pyplot as plt


height = [1.82,  1.87, 1.82, 1.91, 1.90, 1.85]
weight = [81, 97.52, 95.25, 92.98, 86.18, 88.45]

np_height = np.array(height)
np_weight = np.array(weight)

bmi = np_weight / (np_height ** 2)

# Print the result
print(bmi)

print(bmi > 23)

print(bmi[bmi > 26])

# matrix mul
a = np.array([[1.0, 2.0], [3.0, 4.0]])
v = np.array([[1.0], [4.0]])
print(a)
print(v)

print(np.matmul(a,v))


print( a[1,:] ) # second row


#svd
u, s, vh = np.linalg.svd(a)
print( np.allclose(a, np.dot(u * s, vh)) )




