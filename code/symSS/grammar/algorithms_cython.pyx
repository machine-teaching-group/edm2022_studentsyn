cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float 
ctypedef np.float_t DTYPE_t
@cython.boundscheck(False) 
@cython.wraparound(False) 
cpdef inside_algorithm(
    np.ndarray[np.int_t, ndim = 1] words, 
    int len_nts, 
    np.ndarray[DTYPE_t, ndim = 2] unary_rules, 
    np.ndarray[DTYPE_t, ndim = 2] binary_rules): 
    '''
        words :: [str], a sentence to parse 
        grammar G with  
            binary_rules : np.array([A_idx,'B_idx,C_idx,prob])
            unary_rules : [(A_idx,str)]
            len_nts : the number of non-terminals

        return inside_probs[V,i,j] : probability of 
            starting from nonterminal V, 
            generating [w_i ... w_j]
            where i == j means w_i

    '''
    cdef int len_w = words.shape[0]
    cdef int len_u = unary_rules.shape[0]
    cdef int len_b = binary_rules.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] inside_probs = np.zeros(
        [len_nts, len_w, len_w], dtype=DTYPE
        )
    cdef int i, j,l, k, bb, u, b, c, d, u1
    for i in range(0,len_w):
        for j in range(0,len_u) :
            u1 = int(unary_rules[j,1])
            if  u1 == words[i]:  
                u = int(unary_rules[j,0])
                inside_probs[u,i,i] += unary_rules[j,2]

    for l in range(1,len_w): 
        # start point 
        for i in range(0,len_w-l):
            #end_point 
            k = i + l 
            # mid point
            for j in range(i, k):
                for bb in range(len_b): 
                    b = int(binary_rules[bb,0])
                    c = int(binary_rules[bb,1])
                    d = int(binary_rules[bb,2])

                    inside_probs[b,i,k] += binary_rules[bb,3]*\
                    inside_probs[c,i,j]*\
                    inside_probs[d,j+1,k]   
    return inside_probs


@cython.boundscheck(False) 
@cython.wraparound(False)  
def viterbi(
    np.ndarray[np.int_t, ndim = 1] words, 
    int len_nts, 
    np.ndarray[DTYPE_t, ndim = 2] unary_rules, 
    np.ndarray[DTYPE_t, ndim = 2] binary_rules): 
    '''
        words :: [str], a sentence to parse 
        grammar G with  
            binary_rules : np.array([A_idx,'B_idx,C_idx,prob])
            unary_rules : [(A_idx,str)]
            len_nts : the number of non-terminals

        return inside_probs[V,i,j] : probability of 
            starting from nonterminal V, 
            generating [w_i ... w_j]
            where i == j means w_i

    '''
    cdef int len_w = words.shape[0]
    cdef int len_u = unary_rules.shape[0]
    cdef int len_b = binary_rules.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] inside_probs = np.zeros(
        [len_nts, len_w, len_w], dtype=DTYPE
        )
    cdef np.ndarray[DTYPE_t, ndim=3] max_len = np.zeros(
        [len_nts, len_w, len_w], dtype=DTYPE
        )
    cdef np.ndarray[DTYPE_t, ndim=3] left = np.zeros(
        [len_nts, len_w, len_w], dtype=DTYPE
        )
    cdef np.ndarray[DTYPE_t, ndim=3] right = np.zeros(
        [len_nts, len_w, len_w], dtype=DTYPE
        )
    cdef np.ndarray[DTYPE_t, ndim=3] split = np.zeros(
        [len_nts, len_w, len_w], dtype=DTYPE
        )



    cdef int i, j,l, k, bb, u, b, c, d , u1
    cdef DTYPE_t v1, v2  
    for i in range(0,len_w):
        for j in range(0,len_u) :
            u1 = int(unary_rules[j,1])
            if u1 == words[i]:  
                u = int(unary_rules[j,0])
                inside_probs[u,i,i] = unary_rules[j,2]
                max_len[u,i,i] = 1 
                left[u,i,i] = words[i]
                right[u,i,i] = -1
                split[u,i,i] = -1                
                
    for l in range(1,len_w): 
        # start point 
        for i in range(0,len_w-l):
            #end_point 
            k = i + l 
            # mid point
            for j in range(i, k):
                for bb in range(len_b): 
                    b = int(binary_rules[bb,0])
                    c = int(binary_rules[bb,1])
                    d = int(binary_rules[bb,2])


                    v1 = inside_probs[b,i,k]
                    v2 = binary_rules[bb,3]*\
                        inside_probs[c,i,j]*\
                        inside_probs[d,j+1,k]   
                    if v2 > v1 : 
                        max_len[b,i,k] = max_len[c,i,j] + max_len[d, j+1, k]
                        inside_probs[b,i,k] = v2
                        left[b,i,k] = c
                        right[b,i,k] = d
                        split[b,i,k] = j

    return inside_probs, max_len, left, right, split 

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def inside_outside(
    np.ndarray[np.int_t, ndim = 1] words, 
    int len_nts, 
    np.ndarray[DTYPE_t, ndim = 2] unary_rules, 
    np.ndarray[DTYPE_t, ndim = 2] binary_rules):     

    cdef int len_w = words.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] A

    A = inside_algorithm(words, len_nts, unary_rules, binary_rules)

    cdef float Z

    Z = A[0,0,len_w-1] 

    if Z == 0 : 
        return None 

    cdef int len_u = unary_rules.shape[0]
    cdef int len_b = binary_rules.shape[0]
    
    cdef np.ndarray[DTYPE_t, ndim=1] exp_count_b = np.zeros(
        [len_b], dtype=DTYPE
        )
    cdef np.ndarray[DTYPE_t, ndim=1] exp_count_u = np.zeros(
        [len_u], dtype=DTYPE
        )
    cdef np.ndarray[DTYPE_t, ndim=3] B = np.zeros(
        [len_nts, len_w, len_w], dtype=DTYPE
        )

    B[0,0,len_w-1] = 1
    cdef int l, i, k, j
    cdef int b0,b1,b2,u0,u1, idx 
    cdef DTYPE_t b3, u2
    for l in range(len_w-1, 0,-1): # wide to narrow 
        # start_point 
        for i in range(0,len_w - l):
            k = i + l # end point
            for j in range(i,k):
                for idx in range(len_b) : 
                    b0 = int(binary_rules[idx,0])
                    b1 = int(binary_rules[idx,1])
                    b2 = int(binary_rules[idx,2])
                    b3 = binary_rules[idx,3]
                    exp_count_b[idx] += B[b0,i,k]*A[b1,i,j]*A[b2,j+1,k]
                    B[b1,i,j] += b3*B[b0,i,k]*A[b2,j+1,k]
                    B[b2,j+1,k] += b3*A[b1,i,j]*B[b0,i,k]

    for i in range(len_w) : 
        for idx in range(len_u): 
            
            u0 = int(unary_rules[idx,0])
            u1 = int(unary_rules[idx,1])

            if u1 == words[i]:
                exp_count_u[idx] += B[u0,i,i]
    for idx in range(len_u): 
        u2 = unary_rules[idx,2]
        exp_count_u[idx] = 1/Z * exp_count_u[idx]*u2

    for idx in range(len_b): 
        b3 = binary_rules[idx,3]
        exp_count_b[idx] = 1/Z * exp_count_b[idx]*b3

    return exp_count_u, exp_count_b

