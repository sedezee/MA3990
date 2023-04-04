import numpy as np 

def gmres(A, b, x0, n_iter):
    # Initialization
    n = x0.shape[0]
    
    H = np.zeros((n_iter + 1, n_iter))
    r0 = b - (A@x0)
    beta = np.linalg.norm(r0)
    Q = np.zeros((n_iter + 1, n))
    qn = r0/beta
    Q[0] = qn

    for j in range(n_iter):
        # Next Krylov vector
        Q[j+1] = A@Q[j]

        # Riff on the Arnoldi 
        for i in range(j + 1):
            H[i, j] =  Q[i].T @ Q[j+1]
            Q[j+1] = Q[j+1] -  (H[i, j] * Q[i])
        H[j + 1, j] = np.linalg.norm(Q[j+1])

        # Add vector to basis
        Q[j + 1] = Q[j+1] / H[j + 1, j]

    # Find approximation in the basis Q
    e1 = np.zeros(n_iter + 1)
    e1[0] = beta
    y = np.linalg.lstsq(H[:j+2, :j+1], e1, rcond=None)[0]

    # Convert back to full basis
    x_new = x0 + Q[:-1].T @ y
    return x_new

## FUNCTION CALLS
print("## BASIC SOLVE WITHOUT PIVOT: \n")
A = np.array([[7,-2,1], [14,-7,-3], [-7,11,18]])
b = np.array([12,17,5])

print(f"A: \n{A}\n")
print(f"B: \n{b}\n")

## EXPECTED RESULTS
print("\n### EXPECTED RESULTS: ")
x = np.linalg.solve(A,b)
print(f"{x}")

print("\n### ACTUAL RESULTS: ")
x = gmres(A, b, np.zeros(3), 20)
print(f"{x}")