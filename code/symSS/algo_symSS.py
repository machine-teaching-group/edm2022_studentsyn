import numpy as np 
import os 

from .grammar.pcfg_cnf import GrammarCNF
from .symbolic_gen.symSS_hoc4 import GenGrammarHoc4, student_types_hoc4
from .symbolic_gen.symSS_hoc18 import GenGrammarHoc18, student_types_hoc18
from code.utils.convert import token_list_to_benchhmark_ast
from code.utils.parser.utils import hoc4_tokens, hoc18_tokens


class SymSS:        

    def observe_task(self, task_A): 

        task_name = task_A["name"]

        if task_name == 'hoc4':
            self.grammar_generator = GenGrammarHoc4() 
            self.terminals = hoc4_tokens
            self.student_types = student_types_hoc4

        elif task_name == 'hoc18' : 
            self.grammar_generator = GenGrammarHoc18() 
            self.terminals = hoc18_tokens
            self.student_types = student_types_hoc18
 
        self.task_A = task_A
        
    def observe_stu(self, stu_A):
        self.stu_A = stu_A
        scores = []
        stu_types = []
        grammars = []

        self.task_grammars_for_stu = {}

        for stu_type in self.student_types:
            grammar_str = self.grammar_generator.generate_grammar(stu_type, self.task_A)
            grammar = GrammarCNF(grammar_str, self.terminals)
            score = self.score_grammar(grammar, stu_A)
            scores.append(score)
            stu_types.append(stu_type)
            grammars.append(grammar)      

        self.grammars = grammars 
        self.stu_types = stu_types

        chosen_idx = np.argmax(scores)
 
        self.chosen_type = stu_types[chosen_idx]
        self.task_grammars_for_stu[self.task_A["name"]] = grammars[chosen_idx]

    def score(self, task_B, options):

        grammar = self.get_grammar_for_task(task_B)
        scores = []

        for opt in options :
            score = self.score_grammar(grammar, opt)
            scores.append(score)

        pred = np.argmax(scores)

        return pred

    def score_grammar(self, grammar, code_ast):

        prob, length = grammar.score(code_ast) 
        length = max(length,1)

        if prob == 0 : prob_norm = float('-inf')
        else : prob_norm = np.log(prob) / length

        return prob_norm

    def generate(self, task_B, top_k = 1, iterations = 1000): 
        
        generations = []
        grammar = self.get_grammar_for_task(task_B)
        gen_tokens, probs = grammar.generate_topk(top_k = top_k, iterations = iterations)

        for gen in gen_tokens:
            token_list = ["DEF","run","m("] + gen + ["m)"]
            gen_ast = token_list_to_benchhmark_ast(token_list)
            generations.append(gen_ast)
    
        return generations, probs


    def get_grammar_for_task(self, task):

        task_name = task["name"]
        if task_name not in self.task_grammars_for_stu : 
            grammar_str = self.grammar_generator.generate_grammar(self.chosen_type, task)
            grammar = GrammarCNF(grammar_str, self.terminals)
            self.task_grammars_for_stu[task_name] = grammar
        else : 
            grammar = self.task_grammars_for_stu[task_name]

        return grammar        

    def save_stu_grammars(self, save_folder):        
        save_folder = save_folder + '/grammars/'
        os.makedirs(save_folder, exist_ok = True)
        for task_name, grammar in self.task_grammars_for_stu.items():
            with open(save_folder + f'grammar_{task_name}.cfg', 'w+') as f:
                f.write(grammar.tostr())
