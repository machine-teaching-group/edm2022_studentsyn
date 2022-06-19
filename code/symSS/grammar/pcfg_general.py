import logging 
import heapq
from token import COMMENT
import numpy as np 
from code.utils.utils import * 


class PCFGProduction:
    '''
        class for PCFGproduction rules 
        of the form : left_side -> right_side ([prob])
        if [prob] is omitted a prob of [1.0] 
        is assumed 
        `#` denotes comments 
    '''
    def __init__(self, left_side, right_side, prob, idx) :
        self.left_side = left_side 
        self.right_side = right_side 
        self.prob = prob  
        self.idx = idx
    def __repr__(self): 
        return f"{self.left_side} -> {self.right_side} ({self.prob})"


class PCFG:
    
    def __repr__(self): 
        myrep = '[\n'
        for p in self.productions(): 
            myrep += '\t' + p.__repr__() + '\n'
        myrep += ']'
        return myrep 

    def to_str(self): 
        my_str = '\n'
        for p in self.productions():
            p_rep = f'{p.left_side} ->'
            for r in p.right_side : 
                p_rep += f' {r}'
            my_str += p_rep + f' [{round(p.prob, 6)}]'
            my_str += '\n'
        
        return my_str

    def __init__(self,grammar_str):
        '''
            load grammar from an str file         
        '''
        rule_lines = grammar_str.split('\n')
        self.productions_ = []
        prod_idx = 0
        self.terminals = set()
        self.non_terminals = set()
        self.productions_NT = {}
        check_unique = set()


        for idx, line in enumerate(rule_lines) : 
            line = line.strip()
            if line == '' : 
                continue 

            if line.startswith('#'):
                continue

            left_side, rest = line.split('->')
            left_side = left_side.strip()

            if left_side not in self.non_terminals : 
                self.productions_NT[left_side] = []
                self.non_terminals.add(left_side)


            for right_side in rest.split('|'):
                if right_side.strip() == '' :
                    print('empty rules not allowed')
                    exit(1)

                right_side = right_side.split()
                if '[' in right_side[-1] :
                    prob = float(right_side[-1][1:-1])
                    right_side = right_side[:-1]
                else : 
                    prob = 1.0


                for token in right_side : 
                    if token[0] in ['\'','\"'] and token[-1] in ['\"','\'']:
                        self.terminals.add(token)
                    else : 
                        if token not in self.non_terminals : 
                            self.productions_NT[token] = []
                        self.non_terminals.add(token)
                code = left_side+''.join(right_side)
                if code in check_unique : 
                    print(code)
                    print("Double production.")
                    continue 

                check_unique.add(code)
                prod = PCFGProduction(left_side,right_side,prob,prod_idx)
                self.productions_.append(prod)
                prod_idx += 1
                
                self.productions_NT[left_side].append(prod)

        self.num_prod = prod_idx

    def productions(self, start = None):
        '''
            get production starting from `start` non-terminal if given 
            else get all productions
        '''
        if start is not None : 
            rets = []
            for p in self.productions_ : 
                if p.left_side == start : 
                    rets.append(p)
            return rets
        else : 
            return self.productions_

    def get_probs(self):
        probs = np.zeros(self.num_prod)
        for prod in self.productions() : 
            probs[prod.idx] = prod.prob            
        return probs 

    def update_probs(self, probs):
        for prob, prod in zip(probs, self.productions()) : 
            prod.prob = prob 


    def normalize(self):
        
        for A,v in self.productions_NT.items() : 
            probsum = sum([max(prod.prob,0) for prod in v])
            if probsum == 0 : probsum = 1 
            for prod in v : 
                prod.prob = max(prod.prob,0) / probsum 

    def generate_topk(self,top_k = 2, iterations = 1000):

        unique_priority_queue = PrioritySet()
        for _ in range(iterations):
            generated_seq, prob, productions = self.generate(normalize = True)
            generated_seq_str = ' '.join(generated_seq)
            key = -prob
            unique_priority_queue.add(generated_seq_str, key)

        predictions = []
        for _ in range(top_k):
            if not unique_priority_queue.is_empty() :
                pred = unique_priority_queue.get()
                predictions.append(pred)
            else : 
                break 
        probs_all =     [ -pred[0] for pred in predictions ]
        sequences_all = [  pred[1].split() for pred in predictions ]

        return sequences_all, probs_all 


    def generate(self, items = [], depth = 0, max_depth = 100, normalize = False):
        if depth == max_depth : 
            print('Warning -- reached max depth, returning "empty"')
            return ['empty'], [1.0], [(self.productions('eps'), [])]

        generation_start = False

        if items == [] : 
            generation_start = True 
            items = ['S']

        list_of_terminals = []
        probs_all = []
        chosen_productions = []

        for token in items: 
            if token in self.terminals:
                terminal_stripped = token.strip('"').strip("'")
                list_of_terminals = [terminal_stripped]
            else : 

                productions = self.productions(token)
                probs_productions = [p.prob for p in productions]
                if probs_productions == [] : 
                    print(f'Error -- no productions for {token}')
                    exit(1)

                probs_productions = np.array(probs_productions)  /  np.sum(probs_productions)

                chosen_production = np.random.choice(productions, p =  probs_productions)
                right_hand_side = chosen_production.right_side
                child_terminals, child_prob , child_productions = self.generate(right_hand_side, depth + 1 , max_depth, normalize)

                chosen_prob = probs_productions[np.where(np.array(productions) == chosen_production)]
                chosen_productions.append((chosen_production, child_productions))
                list_of_terminals += child_terminals


                # need also to add probs for non-term transitions
                probs_all += chosen_prob.tolist()
                probs_all += child_prob

     
        terminal_sequence = list_of_terminals
        probs_all = probs_all

        if generation_start : 
            # remove empty
            terminal_sequence = list(filter(lambda x : x != 'empty',terminal_sequence))
            tree_length = len(probs_all)                 
            probs_all = np.sum(np.log(probs_all))
            if normalize : 
                probs_all = probs_all / tree_length
            probs_all = np.exp(probs_all)

        return terminal_sequence, probs_all, chosen_productions


class PrioritySet(object):
    def __init__(self):
        self.heap = []
        self.set = set()

    def is_empty(self): 
        return len(self.set) == 0

    def add(self, d, pri):
        if not d in self.set:
            heapq.heappush(self.heap, (pri, d))
            self.set.add(d)

    def get(self):
        pri, d = heapq.heappop(self.heap)
        self.set.remove(d)
        return (pri, d)


