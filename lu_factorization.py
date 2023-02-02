import numpy as np

# calculate LU factors
# TODO : check if pivoting is needed (part of implementing PLU)
def __lu_factor(A):
    n = len(A) 
    L = np.identity(n)
    U = A
    for i in range(n): 
        for j in range(i+1, n):
            L[j, i] = U[j,i] / U[i,i]
            U[j] = (U[j] - (L[j,i] * U[i]))
    
    return L, U

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
        [L, U] = __lu_factor(A) # if not, run the LU_Factor method and solve for y, then x 
        y = solve(L, b)
        x = solve(U, y)
    return x 

## FUNCTION CALLS

print("LU CALLS: ")
A = np.array([[7,-2,1], [14,-7,-3], [-7,11,18]])
b = np.array([12,17,5])
[L,U] = __lu_factor(A)
print(f"L: \n{L}\n")
print(f"U: \n{U}\n")

## EXPECTED RESULTS
print("\nEXPECTED RESULTS: ")
# Ly = b
y = np.linalg.inv(L).dot(b)
print(f"y: {y}\n")

# Ux = y
x = np.linalg.inv(U).dot(y)
print(f"x: {x}")

# alternatively, by hand...
print("\nHARDCODED SOLVE WITH PRIOR LU\n")

# Ly = b

y = solve(L,  b)
print(f"y: {y}\n")

# Ux = y

x = solve(U, y)
print(f"x: {x}")

print("\nFULL SOLVE:")
print(f"x: {solve(A,b)}")