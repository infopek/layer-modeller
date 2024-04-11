__device__ void swap(size_t& a, size_t& b) {
    size_t temp = a;
    a = b;
    b = temp;
}

__device__ void swap(double& a, double& b) {
    double temp = a;
    a = b;
    b = temp;
}

// INPUT: A - array of pointers to rows of a square matrix having dimension N
//        Tol - small tolerance number to detect failure when the matrix is near degenerate
// OUTPUT: Matrix A is changed, it contains a copy of both matrices L-E and U as A=(L-E)+U such that P*A=L*U.
//        The permutation matrix is not stored as a matrix, but in an integer vector P of size N+1 
//        containing column indexes where the permutation matrix has "1". The last element P[N]=S+N, 
//        where S is the number of row exchanges needed for determinant computation, det(P)=(-1)^S    
__device__ size_t LUPDecompose(double *A, size_t N, double Tol, size_t *P) {

    size_t i, j, k, imax;
    double maxA, *ptr, absA;

    for (i = 0; i <= N; i++)
        P[i] = i; // Unit permutation matrix, P[N] initialized with N

    for (i = 0; i < N; i++) {
        maxA = 0.0f;
        imax = i;

        for (k = i; k < N; k++)
            if ((absA = fabsf(A[k * N + i])) > maxA) {
                maxA = absA;
                imax = k;
            }

        if (maxA < Tol)
            return 0; // failure, matrix is degenerate

        if (imax != i) {
            // pivoting P
            swap(P[i], P[imax]);

            // pivoting rows of A
            for (j = 0; j < N; j++)
                swap(A[i * N + j], A[imax * N + j]);

            // counting pivots starting from N (for determinant)
            P[N]++;
        }

        for (j = i + 1; j < N; j++) {
            A[j * N + i] /= A[i * N + i];

            for (k = i + 1; k < N; k++)
                A[j * N + k] -= A[j * N + i] * A[i * N + k];
        }
    }

    return 1; // decomposition done
}

// INPUT: A,P filled in LUPDecompose; b - rhs vector; N - dimension
// OUTPUT: x - solution vector of A*x=b
__device__ void LUPSolve(double *A, size_t *P, double *b, size_t N, double *x) {

    for (size_t i = 0; i < N; i++) {
        x[i] = b[P[i]];

        for (size_t k = 0; k < i; k++)
            x[i] -= A[i * N + k] * x[k];
    }

    for (size_t i = N - 1; i >= 0; i--) {
        for (size_t k = i + 1; k < N; k++)
            x[i] -= A[i * N + k] * x[k];

        x[i] /= A[i * N + i];
    }
}

// INPUT: A,P filled in LUPDecompose; N - dimension
// OUTPUT: IA is the inverse of the initial matrix
__device__ void LUPInvert(double *A, size_t *P, size_t N, double *IA) {

    for (size_t j = 0; j < N; j++) {
        for (size_t i = 0; i < N; i++) {
            IA[i * N + j] = P[i] == j ? 1.0f : 0.0f;

            for (size_t k = 0; k < i; k++)
                IA[i * N + j] -= A[i * N + k] * IA[k * N + j];
        }

        for (size_t i = N - 1; i >= 0; i--) {
            for (size_t k = i + 1; k < N; k++)
                IA[i * N + j] -= A[i * N + k] * IA[k * N + j];

            IA[i * N + j] /= A[i * N + i];
        }
    }
}