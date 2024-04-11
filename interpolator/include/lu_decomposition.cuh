__device__ void swap(size_t& a, size_t& b);
__device__ void swap(double& a, double& b);
// INPUT: A - array of pointers to rows of a square matrix having dimension N
//        Tol - small tolerance number to detect failure when the matrix is near degenerate
// OUTPUT: Matrix A is changed, it contains a copy of both matrices L-E and U as A=(L-E)+U such that P*A=L*U.
//        The permutation matrix is not stored as a matrix, but in an integer vector P of size N+1 
//        containing column indexes where the permutation matrix has "1". The last element P[N]=S+N, 
//        where S is the number of row exchanges needed for determinant computation, det(P)=(-1)^S    
__device__ size_t LUPDecompose(double *A, size_t N, double Tol, size_t *P);
// INPUT: A,P filled in LUPDecompose; b - rhs vector; N - dimension
//
// OUTPUT: x - solution vector of A*x=b
__device__ void LUPSolve(double *A, size_t *P, double *b, size_t N, double *x);
// INPUT: A,P filled in LUPDecompose; N - dimension
// OUTPUT: IA is the inverse of the initial matrix
__device__ void LUPInvert(double *A, size_t *P, size_t N, double *IA);