import numpy as np

def arnoldi_iter(A, b, n):
    m = A.shape[0]

    h = np.zeros((n+1, n))
    Q = np.zeros((m, n+1))

    qn = b / np.linalg.norm(b)
    Q[:, 0] = qn

    for i in range(1, n+1):
        v = A @ qn

        for j in range(i):
            h[j, i-1] = Q[:, j].conj().T @ v
            v = v - (h[j, i-1] * Q[:, j])  # doesn't need dot multiplication, because h_{ji} is a scalar

        h[i, i-1] = np.linalg.norm(v)

        if h[i, i-1] > 1e-10:
            qn = v / h[i, i-1]
            Q[:, i] = qn
        else:
            return Q[:, 0:n], h[0:n, :]

    return Q[:, 0:n], h[0:n, :]

A = np.array([[1, 2], [3, 4]])
# A = np.array([[1, 4, 2], [1, 2, 3], [2, 1, 3]])
# A = np.array([[2, 2, 1, 2], [2, 3, 1, 4], [1, 1, 3, 2], [2, 4, 2, 1]])
b = np.array([2, 2])
n = 2
[Q, h] = arnoldi_iter(A, b, n)

w, v = np.linalg.eig(A)
print("Q: ")
print(np.round(Q, decimals = 3))
print("h:")
print(np.rint(h))

print("\n\n")
print("QhQ*:")
QhQ = Q @ h @ Q.conj().T
print(np.rint(QhQ))

print("Q*AQ:")
QAQ = Q.conj().T @ A @ Q
print(np.rint(QAQ))

print("AQ - Qh")
AQmQH = (A@Q) - (Q @ h)
print(np.rint(AQmQH))

print("QQ*")
QQ = Q @ Q.conj().T
print(np.rint(QQ))