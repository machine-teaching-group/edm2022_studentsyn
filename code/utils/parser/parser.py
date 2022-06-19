from .yacc import yacc 
from code.utils.utils import * 
import ply.lex as lex
from .utils import LoopEnd


class Parser:
    cond_to_attr = {
        "bool_path_ahead" : "path_ahead",
        "bool_path_left" : "path_left",
        "bool_path_right" : "path_right"
    }
    tokens = [
            'RUN', 'DEF',
            'M_LBRACE', 'M_RBRACE', 'C_LBRACE', 'C_RBRACE', 'R_LBRACE', 'R_RBRACE',
            'W_LBRACE', 'W_RBRACE', 'I_LBRACE', 'I_RBRACE', 'E_LBRACE', 'E_RBRACE',
            'WHILE', 'REPEAT','INT',
            'IF', 'IFELSE', 'ELSE',
            'PATH_AHEAD', 'PATH_LEFT', 'PATH_RIGHT',
            'MOVE', 'TURN_RIGHT', 'TURN_LEFT','NO_MARKER'
    ]
    t_ignore =' \t\n'

    t_M_LBRACE = 'm\('
    t_M_RBRACE = 'm\)'

    t_C_LBRACE = 'c\('
    t_C_RBRACE = 'c\)'

    t_R_LBRACE = 'r\('
    t_R_RBRACE = 'r\)'

    t_W_LBRACE = 'w\('
    t_W_RBRACE = 'w\)'

    t_I_LBRACE = 'i\('
    t_I_RBRACE = 'i\)'

    t_E_LBRACE = 'e\('
    t_E_RBRACE = 'e\)'

    t_RUN = 'run'
    t_DEF = 'DEF'
    t_WHILE = 'WHILE'
    t_REPEAT = 'REPEAT'
    t_IF = 'IF'
    t_IFELSE = 'IFELSE'
    t_ELSE = 'ELSE'
    t_NO_MARKER = 'bool_no_marker'
    t_PATH_AHEAD = 'bool_path_ahead'
    t_PATH_LEFT = 'bool_path_left'
    t_PATH_RIGHT = 'bool_path_right'

    conditional_functions = [
        t_PATH_AHEAD,t_PATH_LEFT,t_PATH_RIGHT
    ]

    t_MOVE = 'move'
    t_TURN_RIGHT = 'turn_right'
    t_TURN_LEFT = 'turn_left'

    action_functions = [
            t_MOVE,
            t_TURN_RIGHT, t_TURN_LEFT,
    ]

    MAX_LOOP = 100
    simulate_speed = 0.01 

    def __init__(self, min_int = 0, max_int = 19, hit_info = True, debug = False):
        self.debug = debug
        self.min_int = min_int 
        self.max_int = max_int
        self.lexer = lex.lex(module=self)
        self.yacc, self.grammar = yacc(
            module=self,
            debug=False,
            tabmodule="_parsetab",
            with_grammar=True)
  
        self.prodnames = self.grammar.Prodnames  

        if hit_info :
            self.flush_hit_info()
        else : 
            self.hit_info = None

    def flush_hit_info(self):
        self.hit_info = {}
        self.funct_table = {}
        self.counter = 0
        self.symbolic_decisions = []


    # ------------------- lexer ------------------


    INT_PREFIX = 'R='
    def t_INT(self, t):
        r'R=\d+'

        value = int(t.value.replace(self.INT_PREFIX, ''))
        if not (self.min_int <= value <= self.max_int):
            raise Exception(" [!] Out of range ({} ~ {}): `{}`". \
                    format(self.min_int, self.max_int, value))

        t.value = value
        return t

    def random_INT(self):
        return "{}{}".format(
                self.INT_PREFIX,
                self.rng.randint(self.min_int, self.max_int + 1))

    def t_error(self, t):
        t.lexer.skip(1)


    # ----------------- parser ------------

    def p_prog(self, p):
        '''prog : DEF RUN M_LBRACE stmt M_RBRACE'''
        stmt = p[4]
        p[0] = stmt

    def p_stmt(self, p):
        '''stmt : while
                | repeat
                | stmt_stmt
                | action
                | if
                | ifelse
        '''
        fn = p[1]
        p[0] = fn

    def p_stmt_stmt(self, p):
        '''stmt_stmt : stmt stmt
        '''
        stmt1, stmt2 = p[1], p[2]
        def fn():
            stmt1(); stmt2()
        p[0] = fn

    def p_if(self, p):
        '''if : IF C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE
        '''
        cond, stmt = p[3], p[6]
        if self.hit_info is not None:
            true_key = "if_true" + str(self.counter)
            self.hit_info[true_key] = 0
            self.counter += 1

            false_key = "if_false" + str(self.counter)
            self.hit_info[false_key] = 0
            self.counter += 1

            def fn():
                if cond():
                    self.hit_info[true_key] += 1
                    self.symbolic_decisions.append("if1")
                    out = stmt()
                else:
                    self.hit_info[false_key] += 1
                    self.symbolic_decisions.append("if0")
                    out = lambda : None 
                return out
        else : 
            fn = lambda: stmt() if cond() else lambda : None 
        p[0] = fn

    def p_ifelse(self, p):
        '''ifelse : IFELSE C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE ELSE E_LBRACE stmt E_RBRACE
        '''
        cond, stmt1, stmt2 = p[3], p[6], p[10]
        if self.hit_info is not None:
            true_key = "ifelse_true" + str(self.counter)
            self.hit_info[true_key] = 0
            self.counter += 1

            false_key = "ifelse_false" + str(self.counter)
            self.hit_info[false_key] = 0
            self.counter += 1

            def fn():
                if cond():
                    self.hit_info[true_key] += 1
                    self.symbolic_decisions.append("ifel1")
                    out = stmt1()
                else:
                    self.hit_info[false_key] += 1
                    self.symbolic_decisions.append("ifel0")
                    out = stmt2()
                return out
        else :
            fn = lambda: stmt1() if cond() else stmt2()
        p[0] = fn

    def p_while(self, p):
        '''while : WHILE C_LBRACE NO_MARKER C_RBRACE W_LBRACE stmt W_RBRACE
        '''
        cond, stmt = p[3], p[6]
        if self.hit_info is not None:

            true_key = "while_true" + str(self.counter)
            self.hit_info[true_key] = 0
            self.counter += 1  

            def fn():
                cond_fn = getattr(self.exec_env, cond)
                while cond_fn() :
                    if (self.hit_info[true_key] > self.MAX_LOOP): 
                        raise LoopEnd
                    self.hit_info[true_key] += 1
                    self.symbolic_decisions.append("w1")
                    stmt()
                else:
                    self.symbolic_decisions.append("w0")
                    lambda : None 

        else:
            def fn():
                while(cond()):
                    stmt()                
        p[0] = fn

    def p_repeat(self, p):
        '''repeat : REPEAT cste R_LBRACE stmt R_RBRACE
        '''
        cste, stmt = p[2], p[4]
        if self.hit_info : 
            true_key = f"repeat_{cste()}" + str(self.counter)
            self.hit_info[true_key] = 0
            self.counter += 1 

            def fn():
                for _ in range(cste()):
                    self.hit_info[true_key] += 1
                    stmt()
        else : 
            def fn():
                for _ in range(cste()):
                    stmt()            
        p[0] = fn

    def p_cond(self, p):
        '''cond :  PATH_RIGHT
                 | PATH_AHEAD 
                 | PATH_LEFT
        '''
        cond_without_not = p[1]
        if hasattr(self, 'exec_env') : 
            p[0] = lambda : getattr(self.exec_env, self.cond_to_attr[cond_without_not])() 
        else : 
            p[0] = lambda : None

    def p_action(self, p):
        '''action : MOVE 
                  | TURN_RIGHT
                  | TURN_LEFT

        '''
        action = p[1]
        def fn():
            if self.debug : 
                self.exec_env.print()
                print(action)
                self.simulate_speed = 0
                if self.simulate_speed != 0 : 
                    print("\033[A"*(self.exec_env.grid_size+2))
                    import time 
                    time.sleep(self.simulate_speed)
                else : 
                    print("\033[A"*(self.exec_env.grid_size+3))
                    input()
            getattr(self.exec_env, action)()
            self.exec_env.add_trace()

        p[0] = fn

    def p_cste(self, p):
        '''cste : INT
        '''
        value = p[1]
        p[0] = lambda: int(value)


    def p_error(self, p):
        raise SyntaxError


    def set_execution_env(self, exec_env):
        self.exec_env = exec_env

    def parse(self,code) : 
        try : 
            self.yacc.parse(code)
        except SyntaxError: 
            return False 
        return True 
    
