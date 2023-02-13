import numpy as np

# calculate LU factors
# TODO : check if pivoting is needed (part of implementing PLU)
def __lu_factor(A):
    n = len(A) 
    L = np.identity(n)
    U = A
    for i in range(n): 
        # L[i+1:n, i] = U[i+1:n, i] / U[i,i]; 
        # U[j] = (U[i+1:n] = (U[i+1:n] - (L[i+1:n, i] * U[i])))
        for j in range(i+1, n):
            L[j, i] = U[j,i] / U[i,i]
            U[j] = (U[j] - (L[j,i] * U[i]))
    
    return L, U

def __plu_factor(A): 
    n = len(A)
    
    P = np.identity(n)
    L = np.identity(n)
    U = A

    for i in range(n):

        for k in range(i,n):
            if not np.isclose(U[i,i], 0):
                break
            U[[k, k+1]] = U[[k+1, k]] # swaps row U_i with row U_{k+1}
            P[[k, k+1]] = P[[k+1, k]] # swaps row P_i with row P_{k+1}

    return P 

# Forward substitution for lower triangular    
def __forward_solve(A, b):
    n = len(A)
    x = np.zeros(n)

    for i in range(0, n):
        x[i] = b[i]/A[i,i]
        b[i+1:n] = b[i+1:n] - A[i+1:n,i]*x[i]; 
    return x

# Backward substitution for upper triangular 
def __backward_solve(A, b):
    n = len(A)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = b[i]/A[i,i]
        b[0:i] = b[0:i] - A[0:i, i] * x[i]
    return x

# Primary solve method 
def solve(A, b): 
    if np.allclose(A, np.tril(A)): # zero everything above diagonal, compare with A. Are they the same? 
        x = __forward_solve(A, b)   # if they are, forward sub 
    elif np.allclose(A, np.triu(A)): # zero everything below the diagonal, compare with A. Are they the same? 
        x = __backward_solve(A, b) # if they are, backward sub
    else:
        P = __plu_factor(A)
        [L, U] = __lu_factor(np.dot(P, A)) # if not, run the LU_Factor method and solve for y, then x 
        y = solve(L, np.dot(P, b))
        x = solve(U, y)
    return x 

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

## FULL SOLVE 
print("\n### FULL SOLVE:")
print(f"{solve(A,b)}")

print("\n")

## PIVOT TESTS
print("## BASIC SOLVE WITH PIVOT: ")
A = np.array([[3,2,-4],[2,3,3],[5,-3,1]])
b = np.array([3,15,14])

print("\n### EXPECTED RESULTS:")
print(f"\n{np.linalg.solve(A,b)}\n")

print("\n### FULL SOLVE:")
print(f"\n{solve(A,b)}\n")
