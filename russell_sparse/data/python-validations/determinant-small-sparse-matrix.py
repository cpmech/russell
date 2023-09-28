import numpy as np
import scipy.linalg as la

a = np.array([
    [ 2, 3, 0, 0, 0],
    [ 3, 0, 4, 0, 6],
    [ 0,-1,-3, 2, 0],
    [ 0, 0, 1, 0, 0],
    [ 0, 4, 2, 0, 1]
], dtype=float)

print(a)
print(f"determinant(a) = {la.det(a)}")

