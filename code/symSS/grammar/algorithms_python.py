import numpy as np 

def inside_algorithm(words, len_nts, unary_rules, binary_rules): 
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
    len_w = len(words)
    inside_probs = np.zeros((len_nts,len_w, len_w))
    for i in range(0,len_w):
        for u in unary_rules : 
            if int(u[1]) == words[i]:
                inside_probs[int(u[0])][i][i] += u[2]

    for l in range(1,len_w): 
        # start point 
        for i in range(0,len_w-l):
            #end_point 
            k = i + l 
            # mid point
            for j in range(i, k):
                for b in binary_rules : 
                    inside_probs[int(b[0]),i,k] += b[-1]*\
                    inside_probs[int(b[1]),i,j]*\
                    inside_probs[int(b[2]),j+1,k]   
    return inside_probs


def viterbi(words, len_nts, unary_rules, binary_rules): 
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
    len_w = len(words)
    inside_probs = np.zeros((len_nts,len_w, len_w))
    max_len = np.zeros((len_nts,len_w, len_w))
    left = np.zeros((len_nts,len_w, len_w))
    right = np.zeros((len_nts,len_w, len_w))
    split = np.zeros((len_nts,len_w, len_w))

    for i in range(0,len_w):
        for u_idx, u in enumerate(unary_rules) : 
            if int(u[1]) == words[i]:
                inside_probs[int(u[0])][i][i] = u[2]
                max_len[int(u[0])][i][i] = 1 
                left[int(u[0])][i][i] = words[i]
                right[int(u[0])][i][i] = -1
                split[int(u[0])][i][i] = -1

        for l in range(1,len_w): 
            # start point 
            for i in range(0,len_w-l):
                #end_point 
                k = i + l 
                # mid point
                for j in range(i, k):
                    for b_idx, b in enumerate(binary_rules): 
                        v1 = inside_probs[int(b[0]),i,k]
                        v2 = b[-1]*inside_probs[int(b[1]),i,j]*\
                            inside_probs[int(b[2]),j+1,k]   
                        if v2 > v1 : 
                            max_len[int(b[0]),i,k] = max_len[int(b[1]),i,j] + max_len[int(b[2]), j+1, k]
                            inside_probs[int(b[0]),i,k] = v2
                            left[int(b[0]),i,k] = b[1]
                            right[int(b[0]),i,k] = b[2]
                            split[int(b[0]),i,k] = j


    return inside_probs, max_len, left, right, split 


def inside_outside(words, len_nts, unary_rules, binary_rules):
    '''
        given a sentences words : [str]
        Grammar G with : 
            rules and number of not-terminals
        S = len_nts-1

        return expected count of each rule with repsect to the data

    '''    
    binary_rules = binary_rules 
    unary_rules = unary_rules
    len_w = len(words)
    # inside probs 
    A = inside_algorithm(words, len_nts, unary_rules, binary_rules)

    Z = A[0,0,-1] 

    if Z == 0 : 
        print('sentence not parsed')
        return None 
    exp_count_b = np.zeros(len(binary_rules))
    exp_count_u = np.zeros(len(unary_rules))

    B = np.zeros((len_nts, len_w, len_w))
    B[0,0,-1] += 1

    for l in range(len_w-1, 0,-1): # wide to narrow 
        # start_point 
        for i in range(0,len_w - l):
            k = i + l # end point
            for j in range(i,k):
                for idx,b in enumerate(binary_rules) : 
                    exp_count_b[idx] += B[int(b[0]),i,k]*A[int(b[1]),i,j]*A[int(b[2]),j+1,k]
                    B[int(b[1]),i,j] += b[-1]*B[int(b[0]),i,k]*A[int(b[2]),j+1,k]
                    B[int(b[2]),j+1,k] += b[-1]*A[int(b[1]),i,j]*B[int(b[0]),i,k]

    for i in range(len_w) : 
        for idx,u in enumerate(unary_rules): 
            if  int(u[1]) == words[i]:
                exp_count_u[idx] += B[int(u[0]),i,i]
    for idx,u in enumerate(unary_rules):
        exp_count_u[idx] = 1/Z * exp_count_u[idx]*u[-1]
    for idx,b in enumerate(binary_rules):
        exp_count_b[idx] = 1/Z * exp_count_b[idx]*b[-1]
    return exp_count_u, exp_count_b