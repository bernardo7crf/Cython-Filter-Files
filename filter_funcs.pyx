
# distutils: language = c
# distutils: sources = interpolation.c
# cython: language_level=3
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
    
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport log, exp, log10 
from libc.stdlib cimport malloc, free
import scipy.fft 
 
 
cdef extern from "interpolation.h" nogil:
    int interpolate(double x[], double y[], double x_interp[], double y_interp[], int num_x, int num_x_interp, int type)

cpdef np.ndarray[np.float64_t, ndim=2] interp1d_openmp(np.ndarray[np.float64_t, ndim=1] k, np.ndarray[np.float64_t, ndim=2] pkz, np.ndarray[np.float64_t, ndim=1] k_interp, int type):
    cdef int num_k = k.shape[0]
    cdef int num_k_interp = k_interp.shape[0]
    cdef int num_z = pkz.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] pkz_interp = np.empty((num_z, num_k_interp))
    cdef int j

    for j in prange(num_z, nogil=True):
        interpolate(&k[0], &pkz[j, 0], &k_interp[0], &pkz_interp[j, 0], num_k, num_k_interp, type)

    return pkz_interp


cpdef np.ndarray[np.float64_t, ndim=2] second_derivative(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=2] y):
    cdef int num_x = x.shape[0]
    cdef int num_y = y.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] d2y_dx2 = np.empty((num_y, num_x))
    cdef int i, j

    for i in prange(num_y, nogil=True):
        d2y_dx2[i, 0] = (2*y[i, 0] - 5*y[i, 1] + 4*y[i, 2] - y[i, 3]) / (x[1] - x[0])**2
        for j in range(1, num_x-1):
            d2y_dx2[i, j] = (y[i, j+1] - 2*y[i, j] + y[i, j-1]) / (x[1] - x[0])**2
        d2y_dx2[i, -1] = (2*y[i, -1] - 5*y[i, -2] + 4*y[i, -3] - y[i, -4]) / (x[1] - x[0])**2

    return d2y_dx2
    
cpdef np.ndarray[np.float64_t, ndim=2] logkpk_openmp(np.ndarray[np.float64_t, ndim=1] dst_ks, np.ndarray[np.float64_t, ndim=2] inter_log_pkz):
    cdef int num_k = dst_ks.shape[0]
    cdef int num_z = inter_log_pkz.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] logkpk = np.empty((num_z, num_k))
    cdef int i, j

    for i in prange(num_z, nogil=True):
        for j in range(num_k):
            logkpk[i, j] = log10(dst_ks[j] * exp(inter_log_pkz[i, j]))

    return logkpk
    
cpdef np.ndarray[np.float64_t, ndim=2] compute_pk_nw(np.ndarray[np.float64_t, ndim=2] treated_logkpk, np.ndarray[np.float64_t, ndim=1] dst_ks):
    cdef int i, j
    cdef int n = treated_logkpk.shape[0]
    cdef int m = treated_logkpk.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] pk_nw = np.empty((n, m), dtype=np.float64)
    for i in prange(n, nogil=True):
        for j in range(m):
            pk_nw[i, j] = 10**(treated_logkpk[i, j])/dst_ks[j]
    return pk_nw
    
    
#PRECISEI TRANSFORMAR [:] to [:,:] PARA ACABAR COM RACE CONDITION 
cpdef np.ndarray[np.float64_t, ndim=2] pk_nw_(np.ndarray[double, ndim=1] even_is,
                np.ndarray[double, ndim=1] odd_is,
                np.ndarray[int, ndim=1] imin_even,
                np.ndarray[int, ndim=1] imin_odd,
                np.ndarray[int, ndim=1] imax_even,
                np.ndarray[int, ndim=1] imax_odd,
                np.ndarray[double, ndim=2] evens,
                np.ndarray[double, ndim=2] odds,
                np.ndarray[double, ndim=1] dst_ks,
                int size_pk):
    cdef int i, j, evenis_index, oddis_index, len_even_is, len_odd_is
    cdef np.ndarray[double, ndim=2] treated_transform = np.empty((size_pk, len(odd_is) + len(even_is)), dtype=np.float64)
    cdef double[:,:] even_is_removed_bumps = np.empty((size_pk, len(even_is)), dtype=np.float64)
    cdef double[:,:] odd_is_removed_bumps = np.empty((size_pk, len(odd_is)), dtype=np.float64)
    cdef double[:,:] evens_removed_bumps = np.empty((size_pk, evens.shape[1]), dtype=np.float64)
    cdef double[:,:] odds_removed_bumps = np.empty((size_pk, odds.shape[1]), dtype=np.float64)
    cdef double[:,:] y_interp_even = np.empty((size_pk, len(even_is)), dtype=np.float64)
    cdef double[:,:] y_interp_odd = np.empty((size_pk, len(odd_is)), dtype=np.float64)
    cdef np.ndarray[double, ndim=2] treated_logkpk
    cdef np.ndarray[double, ndim=2] pk_nw
    
    len_even_is = len(even_is)
    len_odd_is = len(odd_is)

    for i in prange(size_pk, nogil=True):

        for j in range(imin_even[i]):
            even_is_removed_bumps[i, j] = even_is[j]
            evens_removed_bumps[i, j] = evens[i, j]
        for j in range(imax_even[i], len_even_is):
            even_is_removed_bumps[i, j-imax_even[i]+imin_even[i]] = even_is[j]
            evens_removed_bumps[i, j-imax_even[i]+imin_even[i]] = evens[i, j]

        for j in range(imin_odd[i]):
            odd_is_removed_bumps[i, j] = odd_is[j]
            odds_removed_bumps[i, j] = odds[i, j]           
        for j in range(imax_odd[i], len_odd_is):
            odd_is_removed_bumps[i, j-imax_odd[i]+imin_odd[i]] = odd_is[j]
            odds_removed_bumps[i, j-imax_odd[i]+imin_odd[i]] = odds[i, j]

 


        evenis_index = len_even_is - imax_even[i] + imin_even[i]
        oddis_index = len_odd_is - imax_odd[i] + imin_odd[i]

        for j in range(evenis_index):
            evens_removed_bumps[i, j] *= (even_is_removed_bumps[i, j] + 1)**2
        for j in range(oddis_index):
            odds_removed_bumps[i, j] *= (odd_is_removed_bumps[i, j] + 1)**2

        interpolate(&even_is_removed_bumps[i, 0], &evens_removed_bumps[i, 0], &even_is[0],
        &y_interp_even[i, 0], evenis_index, len_even_is, 1)
        interpolate(&odd_is_removed_bumps[i, 0], &odds_removed_bumps[i, 0], &odd_is[0],
        &y_interp_odd[i, 0], oddis_index, len_odd_is, 1)

        for j in range(len_even_is):
            treated_transform[i, j*2] = y_interp_even[i, j]/(even_is[j] + 1)**2
        for j in range(len_odd_is):
            treated_transform[i, j*2+1] = y_interp_odd[i, j]/(odd_is[j] + 1)**2
    
    treated_logkpk = scipy.fft.idst(treated_transform,
                                    type=2,
                                    workers=-1,
                                    norm = 'forward')/(2*dst_ks.shape[0])
    pk_nw=compute_pk_nw(treated_logkpk, dst_ks)
    return pk_nw    

cpdef np.ndarray[np.float64_t, ndim=2] pksmooth_openmp(double[:, :] pk_nw, 
                                                               double[:, :] p_highk, 
                                                               double[:] k_extended, 
                                                               double[:] ks, 
                                                               double[:] dst_ks, 
                                                               int num_bins_smaller):
    cdef int i, j
    cdef int n = pk_nw.shape[0]
    cdef int m = k_extended.shape[0]
    cdef int p = ks.shape[0] - num_bins_smaller
    cdef int p_highk_len = p_highk.shape[1]
    cdef int p_extended_len
    cdef double[:, :] pksmooth_interp = np.empty((n, p), dtype=np.float64)
    cdef double[:] log_k_extended = np.log(k_extended)
    cdef double[:] x_interp = np.log(ks[num_bins_smaller:])
    cdef double[:, :] y_interp
    cdef double[:, :] p_extended_np
    cdef double[:,:] log_p_extended
    
    cdef int count = 0
    for j in range(dst_ks.shape[0]):
        if dst_ks[j] < 4:
            count += 1
    
    p_extended_len = count + p_highk_len
    p_extended_np = np.empty((n, p_extended_len), dtype=np.float64)
    log_p_extended = p_extended_np
    y_interp = np.empty((n, p), dtype=np.float64)
    
    with nogil:
        for i in prange(n):
            for j in range(count):
                log_p_extended[i, j] = log(pk_nw[i, j])
            for j in range(count, p_extended_len):
                log_p_extended[i, j] = log(p_highk[i, j - count]) 

            if interpolate(&log_k_extended[0], &log_p_extended[i,0], &x_interp[0], &y_interp[i, 0], p_extended_len, p, 1) != 0:
                with gil:
                    raise ValueError("Interpolation failed")

            for j in range(p):
                pksmooth_interp[i, j] = y_interp[i, j]
            
    return np.asarray(pksmooth_interp)
