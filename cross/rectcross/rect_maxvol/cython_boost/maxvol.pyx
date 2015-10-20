__all__ = ['c_maxvol', 'c_rect_maxvol']
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, realloc, free
from cython.parallel import prange

def c_rect_maxvol(A, tol = 1., maxK = None, min_add_K = None, minK = None, start_maxvol_iters = 10, verbose = False, identity_submatrix = True):
    """Cython implementation of rectangular 2-volume maximization. For information see :py:func:`rect_maxvol` function"""
    cdef int N, r, id_sub
    cdef cnp.ndarray lu, coef, basis
    if type(A) != np.ndarray:
        raise TypeError, "argument must be of numpy.ndarray type"
    if len(A.shape) != 2:
        raise ValueError, "argument must have 2 dimensions"
    N, r = A.shape
    if N <= r:
        return np.arange(N, dtype = np.int32), np.eye(N, dtype = A.dtype)
    lu = np.copy(A, order = 'F')
    if maxK is None or maxK > N:
        maxK = N
    if maxK < r:
        maxK = r
    if minK is None or minK < r:
        minK = r
    if minK > N:
        minK = N
    if min_add_K is not None:
        minK = max(minK, r + min_add_K) 
    if minK > maxK:
        minK = maxK
    if identity_submatrix:
        id_sub = 1
    else:
        id_sub = 0
    try:
        if A.dtype is np.dtype(np.float32):
            return srect_maxvol(N, r, <float *>lu.data, tol, minK, maxK, start_maxvol_iters, id_sub)
        elif A.dtype is np.dtype(np.float64):
            return drect_maxvol(N, r, <double *>lu.data, tol, minK, maxK, start_maxvol_iters, id_sub)
        elif A.dtype is np.dtype(np.complex64):
            return crect_maxvol(N, r, <float complex *>lu.data, tol, minK, maxK, start_maxvol_iters, id_sub)
        elif A.dtype is np.dtype(np.complex128):
            return zrect_maxvol(N, r, <double complex*>lu.data, tol, minK, maxK, start_maxvol_iters, id_sub)
    except Exception:
        raise

def c_maxvol(A, tol = 1.05, max_iters = 100):
    """Cython implementation of 1-volume maximization. For information see :py:func:`maxvol` function"""
    cdef int N, r
    cdef cnp.ndarray lu, coef, basis
    if type(A) != np.ndarray:
        raise TypeError, "argument must be of numpy.ndarray type"
    if len(A.shape) != 2:
        raise ValueError, "argument must have 2 dimensions"
    N, r = A.shape
    if N <= r:
        return np.arange(N, dtype = np.int32), np.eye(N, dtype = A.dtype)
    if tol < 1:
        tol = 1.0
    lu = np.copy(A, order = 'F')
    coef = np.copy(lu, order = 'F')
    basis = np.ndarray(r, dtype = np.int32)
    try:
        if A.dtype is np.dtype(np.float32):
            smaxvol(N, r, <float *>lu.data, <float *>coef.data, <int *>basis.data, tol, max_iters)
        elif A.dtype == np.dtype(np.float64):
            dmaxvol(N, r, <double *>lu.data, <double *>coef.data, <int *>basis.data, tol, max_iters)
        elif A.dtype is np.dtype(np.complex64):
            cmaxvol(N, r, <float complex *>lu.data, <float complex *>coef.data, <int *>basis.data, tol, max_iters)
        elif A.dtype is np.dtype(np.complex128):
            zmaxvol(N, r, <double complex*>lu.data, <double complex *>coef.data, <int *>basis.data, tol, max_iters)
        else:
            raise TypeError("must be of float or complex type")
    except Exception:
        raise
    return basis, coef

cdef extern:
    void sgetrf_(int *, int *, float *, int *, int *, int *)
    void strsm_(char *, char *, char *, char *, int *, int *, float *, float *, int *, float *, int *)
    int isamax_(int *, float *, int *)
    void scopy_(int *, float *, int *, float *, int *)
    void sger_(int *, int *, float *, float *, int *, float *, int *, float *, int *)
    float sdot_(int *, float *, int *, float *, int *)
    void sgemv_(char *, int *, int *, float *, float *, int *, float *, int *, float *, float *, int *)
    void dgetrf_(int *, int *, double *, int *, int *, int *)
    void dtrsm_(char *, char *, char *, char *, int *, int *, double *, double *, int *, double *, int *)
    int idamax_(int *, double *, int *)
    void dcopy_(int *, double *, int *, double *, int *)
    void dger_(int *, int *, double *, double *, int *, double *, int *, double *, int *)
    double ddot_(int *, double *, int *, double *, int *)
    void dgemv_(char *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *)
    void cgetrf_(int *, int *, float complex *, int *, int *, int *)
    void ctrsm_(char *, char *, char *, char *, int *, int *, float complex *, float complex *, int *, float complex *, int *)
    void ccopy_(int *, float complex *, int *, float complex *, int *)
    void cgerc_(int *, int *, float complex *, float complex *, int *, float complex *, int *, float complex *, int *)
    void cgeru_(int *, int *, float complex *, float complex *, int *, float complex *, int *, float complex *, int *)
    float complex cdot_(int *, float complex *, int *, float complex *, int *)
    void cgemv_(char *, int *, int *, float complex *, float complex *, int *, float complex *, int *, float complex *, float complex *, int *)
    void zgetrf_(int *, int *, double complex *, int *, int *, int *)
    void ztrsm_(char *, char *, char *, char *, int *, int *, double complex *, double complex *, int *, double complex *, int *)
    void zcopy_(int *, double complex *, int *, double complex *, int *)
    void zgerc_(int *, int *, double complex *, double complex *, int *, double complex *, int *, double complex *, int *)
    void zgeru_(int *, int *, double complex *, double complex *, int *, double complex *, int *, double complex *, int *)
    double complex zdot_(int *, double complex *, int *, double complex *, int *)
    void zgemv_(char *, int *, int *, double complex *, double complex *, int *, double complex *, int *, double complex *, double complex *, int *)



cdef object srect_maxvol(int N, int R, float *lu, float tol, int minK, int maxK, int start_maxvol_iters, int identity_submatrix):
    cdef char cN = 'N'
    cdef int i, j, i_one = 1, K, size = N*R
    cdef float d_one = 1.0, d_zero = 0.0, l
    cdef float tol2 = tol*tol, tmp
    cdef int *basis = <int *> malloc(N * sizeof(int))
    cdef float *chosen = <float *> malloc(N * sizeof(float))
    cdef int [:]basis_buf
    cdef int coef_realloc_step = R, coef_columns = R+coef_realloc_step
    cdef float *coef = <float *> malloc(N * coef_columns * sizeof(float))
    cdef float *tmp_pointer
    cdef float *L = <float *> malloc(N * sizeof(float))
    cdef float *V = <float *> malloc(N * sizeof(float))
    cdef float *tmp_row = <float *> malloc(N * sizeof(float))
    cdef float [:,:] coef_buf
    scopy_(&size, lu, &i_one, coef, &i_one)
    smaxvol(N, R, lu, coef, basis, tol, start_maxvol_iters)
    # compute square length for each vector
    for j in range(N):
        L[j] = 0.0
        V[j] = 0.0
        chosen[j] = 1.0
    for i in range(R):
        tmp_pointer = coef+i*N
        for j in range(N):
            tmp = abs(tmp_pointer[j])
            L[j] += tmp*tmp
    for i in range(R):
        L[basis[i]] = 0.0
        chosen[basis[i]] = 0.0
    i = isamax_(&N, L, &i_one)-1
    K = R
    while K < minK or (L[i] > tol2 and K < maxK):
        basis[K] = i
        chosen[i] = 0.0
        #scopy_(&K, coef+i, &N, tmp_row, &i_one)
        tmp_pointer = coef+i
        for j in range(K):
            tmp_row[j] = tmp_pointer[j*N]
        sgemv_(&cN, &N, &K, &d_one, coef, &N, tmp_row, &i_one, &d_zero, V, &i_one)
        l = (-d_one)/(1+V[i])
        sger_(&N, &K, &l, V, &i_one, tmp_row, &i_one, coef, &N)
        l = -l
        if coef_columns <= K:
            coef_columns += coef_realloc_step
            coef = <float *> realloc(coef, N * coef_columns * sizeof(float))
        tmp_pointer = coef+K*N
        for j in range(N):
            tmp_pointer[j] = l*V[j]
        for j in range(N):
            tmp = abs(V[j])
            L[j] -= (l*tmp*tmp)
            L[j] *= chosen[j]
        i = isamax_(&N, L, &i_one)-1
        K += 1
    free(L)
    free(V)
    free(tmp_row)
    C = np.ndarray((N, K), order = 'F', dtype = np.float32)
    coef_buf = C
    for i in range(K):
        for j in range(N):
            coef_buf[j, i] = coef[i*N+j]
    free(coef)
    if identity_submatrix == 1:
        for i in range(K):
            tmp_pointer = &coef_buf[0, 0]+basis[i]
            for j in range(K):
                tmp_pointer[j*N] = 0.0
            tmp_pointer[i*N] = 1.0
    I = np.ndarray(K, dtype = np.int32)
    basis_buf = I
    for i in range(K):
        basis_buf[i] = basis[i]
    free(basis)
    return I, C

cdef object smaxvol(int N, int R, float *lu, float *coef, int *basis, float tol, int max_iters):
    cdef int *ipiv = <int *> malloc( R * sizeof(int) )
    cdef int *interchange = <int *> malloc( N * sizeof(int) )
    cdef float *tmp_row = <float *> malloc( R*sizeof(float) )
    cdef float *tmp_column = <float *> malloc( N*sizeof(float) )
    cdef int info = 0, size = N * R, i, j, tmp_int, i_one = 1, iters = 0
    cdef char cR = 'R', cN = 'N', cU = 'U', cL = 'L'
    cdef float d_one = 1, alpha, max_value
    cdef float abs_max, tmp
    if (ipiv == NULL or interchange == NULL or tmp_row == NULL or tmp_column == NULL):
        raise MemoryError, "malloc failed to allocate temporary buffers"
    sgetrf_(&N, &R, lu, &N, ipiv, &info)
    if info < 0:
        raise ValueError, "Internal maxvol_fullrank error, {} argument of sgetrf_ had illegal value".format(info)
    if info > 0:
        raise ValueError, "Input matrix must not be singular"
    for i in prange(N, schedule = "static", nogil = True):
        interchange[i] = i
    for i in range(R):
        j = ipiv[i]-1
        if j != i:
            tmp_int = interchange[i]
            interchange[i] = interchange[j]
            interchange[j] = tmp_int
    free(ipiv)
    for i in prange(R, schedule = "static", nogil = True):
        basis[i] = interchange[i]
    free(interchange)
    strsm_(&cR, &cU, &cN, &cN, &N, &R, &d_one, lu, &N, coef, &N)
    strsm_(&cR, &cL, &cN, &cU, &N, &R, &d_one, lu, &N, coef, &N)
    while iters < max_iters:
        abs_max = -1
        for k in range(size):
            tmp = abs(coef[k])
            if tmp > abs_max:
                abs_max = tmp
                j = k
        max_value = coef[j]
        i = j/N
        j -= i*N
        if abs_max > tol:
            scopy_(&R, coef+j, &N, tmp_row, &i_one)
            tmp_row[i] -= d_one
            scopy_(&N, coef+i*N, &i_one, tmp_column, &i_one)
            basis[i] = j
            alpha = (-d_one)/max_value
            sger_(&N, &R, &alpha, tmp_column, &i_one, tmp_row, &i_one, coef, &N)
            iters += i_one
        else:
            break
    free(tmp_row)
    free(tmp_column)
    return

cdef object drect_maxvol(int N, int R, double *lu, double tol, int minK, int maxK, int start_maxvol_iters, int identity_submatrix):
    cdef char cN = 'N'
    cdef int i, j, i_one = 1, K, size = N*R
    cdef double d_one = 1.0, d_zero = 0.0, l
    cdef double tol2 = tol*tol, tmp
    cdef int *basis = <int *> malloc(N * sizeof(int))
    cdef double *chosen = <double *> malloc(N * sizeof(double))
    cdef int [:]basis_buf
    cdef int coef_realloc_step = R, coef_columns = R+coef_realloc_step
    cdef double *coef = <double *> malloc(N * coef_columns * sizeof(double))
    cdef double *tmp_pointer
    cdef double *L = <double *> malloc(N * sizeof(double))
    cdef double *V = <double *> malloc(N * sizeof(double))
    cdef double *tmp_row = <double *> malloc(N * sizeof(double))
    cdef double [:,:] coef_buf
    dcopy_(&size, lu, &i_one, coef, &i_one)
    dmaxvol(N, R, lu, coef, basis, tol, start_maxvol_iters)
    # compute square length for each vector
    for j in range(N):
        L[j] = 0.0
        V[j] = 0.0
        chosen[j] = 1.0
    for i in range(R):
        tmp_pointer = coef+i*N
        for j in range(N):
            tmp = abs(tmp_pointer[j])
            L[j] += tmp*tmp
    for i in range(R):
        L[basis[i]] = 0.0
        chosen[basis[i]] = 0.0
    i = idamax_(&N, L, &i_one)-1
    K = R
    while K < minK or (L[i] > tol2 and K < maxK):
        basis[K] = i
        chosen[i] = 0.0
        #dcopy_(&K, coef+i, &N, tmp_row, &i_one)
        tmp_pointer = coef+i
        for j in range(K):
            tmp_row[j] = tmp_pointer[j*N]
        dgemv_(&cN, &N, &K, &d_one, coef, &N, tmp_row, &i_one, &d_zero, V, &i_one)
        l = (-d_one)/(1+V[i])
        dger_(&N, &K, &l, V, &i_one, tmp_row, &i_one, coef, &N)
        l = -l
        if coef_columns <= K:
            coef_columns += coef_realloc_step
            coef = <double *> realloc(coef, N * coef_columns * sizeof(double))
        tmp_pointer = coef+K*N
        for j in range(N):
            tmp_pointer[j] = l*V[j]
        for j in range(N):
            tmp = abs(V[j])
            L[j] -= (l*tmp*tmp)
            L[j] *= chosen[j]
        i = idamax_(&N, L, &i_one)-1
        K += 1
    free(L)
    free(V)
    free(tmp_row)
    C = np.ndarray((N, K), order = 'F', dtype = np.float64)
    coef_buf = C
    for i in range(K):
        for j in range(N):
            coef_buf[j, i] = coef[i*N+j]
    free(coef)
    if identity_submatrix == 1:
        for i in range(K):
            tmp_pointer = &coef_buf[0, 0]+basis[i]
            for j in range(K):
                tmp_pointer[j*N] = 0.0
            tmp_pointer[i*N] = 1.0
    I = np.ndarray(K, dtype = np.int32)
    basis_buf = I
    for i in range(K):
        basis_buf[i] = basis[i]
    free(basis)
    return I, C

cdef object dmaxvol(int N, int R, double *lu, double *coef, int *basis, double tol, int max_iters):
    cdef int *ipiv = <int *> malloc( R * sizeof(int) )
    cdef int *interchange = <int *> malloc( N * sizeof(int) )
    cdef double *tmp_row = <double *> malloc( R*sizeof(double) )
    cdef double *tmp_column = <double *> malloc( N*sizeof(double) )
    cdef int info = 0, size = N * R, i, j, tmp_int, i_one = 1, iters = 0
    cdef char cR = 'R', cN = 'N', cU = 'U', cL = 'L'
    cdef double d_one = 1, alpha, max_value
    cdef double abs_max, tmp
    if (ipiv == NULL or interchange == NULL or tmp_row == NULL or tmp_column == NULL):
        raise MemoryError, "malloc failed to allocate temporary buffers"
    dgetrf_(&N, &R, lu, &N, ipiv, &info)
    if info < 0:
        raise ValueError, "Internal maxvol_fullrank error, {} argument of dgetrf_ had illegal value".format(info)
    if info > 0:
        raise ValueError, "Input matrix must not be singular"
    for i in prange(N, schedule = "static", nogil = True):
        interchange[i] = i
    for i in range(R):
        j = ipiv[i]-1
        if j != i:
            tmp_int = interchange[i]
            interchange[i] = interchange[j]
            interchange[j] = tmp_int
    free(ipiv)
    for i in prange(R, schedule = "static", nogil = True):
        basis[i] = interchange[i]
    free(interchange)
    dtrsm_(&cR, &cU, &cN, &cN, &N, &R, &d_one, lu, &N, coef, &N)
    dtrsm_(&cR, &cL, &cN, &cU, &N, &R, &d_one, lu, &N, coef, &N)
    while iters < max_iters:
        abs_max = -1
        for k in range(size):
            tmp = abs(coef[k])
            if tmp > abs_max:
                abs_max = tmp
                j = k
        max_value = coef[j]
        i = j/N
        j -= i*N
        if abs_max > tol:
            dcopy_(&R, coef+j, &N, tmp_row, &i_one)
            tmp_row[i] -= d_one
            dcopy_(&N, coef+i*N, &i_one, tmp_column, &i_one)
            basis[i] = j
            alpha = (-d_one)/max_value
            dger_(&N, &R, &alpha, tmp_column, &i_one, tmp_row, &i_one, coef, &N)
            iters += i_one
        else:
            break
    free(tmp_row)
    free(tmp_column)
    return

cdef object crect_maxvol(int N, int R, float complex *lu, float tol, int minK, int maxK, int start_maxvol_iters, int identity_submatrix):
    cdef char cN = 'N'
    cdef int i, j, i_one = 1, K, size = N*R
    cdef float complex d_one = 1.0, d_zero = 0.0, l
    cdef float tol2 = tol*tol, tmp
    cdef int *basis = <int *> malloc(N * sizeof(int))
    cdef float *chosen = <float *> malloc(N * sizeof(float))
    cdef int [:]basis_buf
    cdef int coef_realloc_step = R, coef_columns = R+coef_realloc_step
    cdef float complex *coef = <float complex *> malloc(N * coef_columns * sizeof(float complex))
    cdef float complex *tmp_pointer
    cdef float *L = <float *> malloc(N * sizeof(float))
    cdef float complex *V = <float complex *> malloc(N * sizeof(float complex))
    cdef float complex *tmp_row = <float complex *> malloc(N * sizeof(float complex))
    cdef float complex [:,:] coef_buf
    ccopy_(&size, lu, &i_one, coef, &i_one)
    cmaxvol(N, R, lu, coef, basis, tol, start_maxvol_iters)
    # compute square length for each vector
    for j in range(N):
        L[j] = 0.0
        V[j] = 0.0
        chosen[j] = 1.0
    for i in range(R):
        tmp_pointer = coef+i*N
        for j in range(N):
            tmp = abs(tmp_pointer[j])
            L[j] += tmp*tmp
    for i in range(R):
        L[basis[i]] = 0.0
        chosen[basis[i]] = 0.0
    i = isamax_(&N, L, &i_one)-1
    K = R
    while K < minK or (L[i] > tol2 and K < maxK):
        basis[K] = i
        chosen[i] = 0.0
        #ccopy_(&K, coef+i, &N, tmp_row, &i_one)
        tmp_pointer = coef+i
        for j in range(K):
            tmp_row[j] = tmp_pointer[j*N].conjugate()
        cgemv_(&cN, &N, &K, &d_one, coef, &N, tmp_row, &i_one, &d_zero, V, &i_one)
        l = (-d_one)/(1+V[i])
        cgerc_(&N, &K, &l, V, &i_one, tmp_row, &i_one, coef, &N)
        l = -l
        if coef_columns <= K:
            coef_columns += coef_realloc_step
            coef = <float complex *> realloc(coef, N * coef_columns * sizeof(float complex))
        tmp_pointer = coef+K*N
        for j in range(N):
            tmp_pointer[j] = l*V[j]
        for j in range(N):
            tmp = abs(V[j])
            L[j] -= (l*tmp*tmp).real
            L[j] *= chosen[j]
        i = isamax_(&N, L, &i_one)-1
        K += 1
    free(L)
    free(V)
    free(tmp_row)
    C = np.ndarray((N, K), order = 'F', dtype = np.complex64)
    coef_buf = C
    for i in range(K):
        for j in range(N):
            coef_buf[j, i] = coef[i*N+j]
    free(coef)
    if identity_submatrix == 1:
        for i in range(K):
            tmp_pointer = &coef_buf[0, 0]+basis[i]
            for j in range(K):
                tmp_pointer[j*N] = 0.0
            tmp_pointer[i*N] = 1.0
    I = np.ndarray(K, dtype = np.int32)
    basis_buf = I
    for i in range(K):
        basis_buf[i] = basis[i]
    free(basis)
    return I, C

cdef object cmaxvol(int N, int R, float complex *lu, float complex *coef, int *basis, float tol, int max_iters):
    cdef int *ipiv = <int *> malloc( R * sizeof(int) )
    cdef int *interchange = <int *> malloc( N * sizeof(int) )
    cdef float complex *tmp_row = <float complex *> malloc( R*sizeof(float complex) )
    cdef float complex *tmp_column = <float complex *> malloc( N*sizeof(float complex) )
    cdef int info = 0, size = N * R, i, j, tmp_int, i_one = 1, iters = 0
    cdef char cR = 'R', cN = 'N', cU = 'U', cL = 'L'
    cdef float complex d_one = 1, alpha, max_value
    cdef float abs_max, tmp
    if (ipiv == NULL or interchange == NULL or tmp_row == NULL or tmp_column == NULL):
        raise MemoryError, "malloc failed to allocate temporary buffers"
    cgetrf_(&N, &R, lu, &N, ipiv, &info)
    if info < 0:
        raise ValueError, "Internal maxvol_fullrank error, {} argument of cgetrf_ had illegal value".format(info)
    if info > 0:
        raise ValueError, "Input matrix must not be singular"
    for i in prange(N, schedule = "static", nogil = True):
        interchange[i] = i
    for i in range(R):
        j = ipiv[i]-1
        if j != i:
            tmp_int = interchange[i]
            interchange[i] = interchange[j]
            interchange[j] = tmp_int
    free(ipiv)
    for i in prange(R, schedule = "static", nogil = True):
        basis[i] = interchange[i]
    free(interchange)
    ctrsm_(&cR, &cU, &cN, &cN, &N, &R, &d_one, lu, &N, coef, &N)
    ctrsm_(&cR, &cL, &cN, &cU, &N, &R, &d_one, lu, &N, coef, &N)
    while iters < max_iters:
        abs_max = -1
        for k in range(size):
            tmp = abs(coef[k])
            if tmp > abs_max:
                abs_max = tmp
                j = k
        max_value = coef[j]
        i = j/N
        j -= i*N
        if abs_max > tol:
            ccopy_(&R, coef+j, &N, tmp_row, &i_one)
            tmp_row[i] -= d_one
            ccopy_(&N, coef+i*N, &i_one, tmp_column, &i_one)
            basis[i] = j
            alpha = (-d_one)/max_value
            cgeru_(&N, &R, &alpha, tmp_column, &i_one, tmp_row, &i_one, coef, &N)
            iters += i_one
        else:
            break
    free(tmp_row)
    free(tmp_column)
    return

cdef object zrect_maxvol(int N, int R, double complex *lu, double tol, int minK, int maxK, int start_maxvol_iters, int identity_submatrix):
    cdef char cN = 'N'
    cdef int i, j, i_one = 1, K, size = N*R
    cdef double complex d_one = 1.0, d_zero = 0.0, l
    cdef double tol2 = tol*tol, tmp
    cdef int *basis = <int *> malloc(N * sizeof(int))
    cdef double *chosen = <double *> malloc(N * sizeof(double))
    cdef int [:]basis_buf
    cdef int coef_realloc_step = R, coef_columns = R+coef_realloc_step
    cdef double complex *coef = <double complex *> malloc(N * coef_columns * sizeof(double complex))
    cdef double complex *tmp_pointer
    cdef double *L = <double *> malloc(N * sizeof(double))
    cdef double complex *V = <double complex *> malloc(N * sizeof(double complex))
    cdef double complex *tmp_row = <double complex *> malloc(N * sizeof(double complex))
    cdef double complex [:,:] coef_buf
    zcopy_(&size, lu, &i_one, coef, &i_one)
    zmaxvol(N, R, lu, coef, basis, tol, start_maxvol_iters)
    # compute square length for each vector
    for j in range(N):
        L[j] = 0.0
        V[j] = 0.0
        chosen[j] = 1.0
    for i in range(R):
        tmp_pointer = coef+i*N
        for j in range(N):
            tmp = abs(tmp_pointer[j])
            L[j] += tmp*tmp
    for i in range(R):
        L[basis[i]] = 0.0
        chosen[basis[i]] = 0.0
    i = idamax_(&N, L, &i_one)-1
    K = R
    while K < minK or (L[i] > tol2 and K < maxK):
        basis[K] = i
        chosen[i] = 0.0
        #zcopy_(&K, coef+i, &N, tmp_row, &i_one)
        tmp_pointer = coef+i
        for j in range(K):
            tmp_row[j] = tmp_pointer[j*N].conjugate()
        zgemv_(&cN, &N, &K, &d_one, coef, &N, tmp_row, &i_one, &d_zero, V, &i_one)
        l = (-d_one)/(1+V[i])
        zgerc_(&N, &K, &l, V, &i_one, tmp_row, &i_one, coef, &N)
        l = -l
        if coef_columns <= K:
            coef_columns += coef_realloc_step
            coef = <double complex *> realloc(coef, N * coef_columns * sizeof(double complex))
        tmp_pointer = coef+K*N
        for j in range(N):
            tmp_pointer[j] = l*V[j]
        for j in range(N):
            tmp = abs(V[j])
            L[j] -= (l*tmp*tmp).real
            L[j] *= chosen[j]
        i = idamax_(&N, L, &i_one)-1
        K += 1
    free(L)
    free(V)
    free(tmp_row)
    C = np.ndarray((N, K), order = 'F', dtype = np.complex128)
    coef_buf = C
    for i in range(K):
        for j in range(N):
            coef_buf[j, i] = coef[i*N+j]
    free(coef)
    if identity_submatrix == 1:
        for i in range(K):
            tmp_pointer = &coef_buf[0, 0]+basis[i]
            for j in range(K):
                tmp_pointer[j*N] = 0.0
            tmp_pointer[i*N] = 1.0
    I = np.ndarray(K, dtype = np.int32)
    basis_buf = I
    for i in range(K):
        basis_buf[i] = basis[i]
    free(basis)
    return I, C

cdef object zmaxvol(int N, int R, double complex *lu, double complex *coef, int *basis, double tol, int max_iters):
    cdef int *ipiv = <int *> malloc( R * sizeof(int) )
    cdef int *interchange = <int *> malloc( N * sizeof(int) )
    cdef double complex *tmp_row = <double complex *> malloc( R*sizeof(double complex) )
    cdef double complex *tmp_column = <double complex *> malloc( N*sizeof(double complex) )
    cdef int info = 0, size = N * R, i, j, tmp_int, i_one = 1, iters = 0
    cdef char cR = 'R', cN = 'N', cU = 'U', cL = 'L'
    cdef double complex d_one = 1, alpha, max_value
    cdef double abs_max, tmp
    if (ipiv == NULL or interchange == NULL or tmp_row == NULL or tmp_column == NULL):
        raise MemoryError, "malloc failed to allocate temporary buffers"
    zgetrf_(&N, &R, lu, &N, ipiv, &info)
    if info < 0:
        raise ValueError, "Internal maxvol_fullrank error, {} argument of zgetrf_ had illegal value".format(info)
    if info > 0:
        raise ValueError, "Input matrix must not be singular"
    for i in prange(N, schedule = "static", nogil = True):
        interchange[i] = i
    for i in range(R):
        j = ipiv[i]-1
        if j != i:
            tmp_int = interchange[i]
            interchange[i] = interchange[j]
            interchange[j] = tmp_int
    free(ipiv)
    for i in prange(R, schedule = "static", nogil = True):
        basis[i] = interchange[i]
    free(interchange)
    ztrsm_(&cR, &cU, &cN, &cN, &N, &R, &d_one, lu, &N, coef, &N)
    ztrsm_(&cR, &cL, &cN, &cU, &N, &R, &d_one, lu, &N, coef, &N)
    while iters < max_iters:
        abs_max = -1
        for k in range(size):
            tmp = abs(coef[k])
            if tmp > abs_max:
                abs_max = tmp
                j = k
        max_value = coef[j]
        i = j/N
        j -= i*N
        if abs_max > tol:
            zcopy_(&R, coef+j, &N, tmp_row, &i_one)
            tmp_row[i] -= d_one
            zcopy_(&N, coef+i*N, &i_one, tmp_column, &i_one)
            basis[i] = j
            alpha = (-d_one)/max_value
            zgeru_(&N, &R, &alpha, tmp_column, &i_one, tmp_row, &i_one, coef, &N)
            iters += i_one
        else:
            break
    free(tmp_row)
    free(tmp_column)
    return
