import numpy as np
from scipy.linalg import hessenberg

def arnoldi_iter(A, b, n):
    m = A.shape[0]
    
    h = np.zeros((n+1, n))
    Q = np.zeros((m, n+1))
    
    q1 = b / np.linalg.norm(b)
    Q[:, 0] = q1

    for i in range(n):
        v = np.dot(A, q1)

        for j in range(i + 1): 
            h[j, i] = np.dot(Q[:, j].conj(), v); 
            v = v - h[j, i] * Q[:, j] # doesn't need dot multiplication, because h_{ji} is a scalar
        
        h[i+1, i] = np.linalg.norm(v)
        
        if h[i + 1, i] > 1e-10:
            q1 = v / h[i + 1, i]
            Q[:, i + 1] = q1
        else:
            return Q, h

    return Q, h

A = np.array([[1, 4, 2], [1, 2, 3],[2, 1, 3]])
A = np.array([[2, 2, 1, 2], [2, 3, 1, 4], [1, 1, 3, 2], [2, 4, 2, 1]])
b = [1, 1, 1, 1]
n = 4
[Q, h] = arnoldi_iter(A, b, n)

w, v = np.linalg.eig(A)
print("Q: ")
print(Q) 
print("\n")
print("h:")
print(h)

print("\n\n")


print(f"Q Len: ")
AxQ = np.dot(A, Q)[:, 0:n]
Qxh = np.dot(Q, h)

print("A*Q:")
print(AxQ)
print("\n")
print("Q * h")
print(Qxh)
print("\n")

print("Equal?")
print(np.allclose(AxQ, Qxh))
