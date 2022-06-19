from .parser import Parser

class ParseToTree(Parser):
    def p_prog(self, p):
        '''prog : DEF RUN M_LBRACE stmt M_RBRACE'''
        stmt = p[4]
        p[0] = {"type" : "run", "children" : stmt}

    def p_stmt(self, p):
        '''stmt : while
                | repeat
                | stmt_stmt
                | action
                | if
                | ifelse
        '''
        if isinstance(p[1], list): 
            p[0] = p[1]
        else : 
            p[0] = [p[1]]

    def p_stmt_stmt(self, p):
        '''stmt_stmt : stmt stmt
        '''
        stmt1, stmt2 = p[1], p[2]
        p[0] = stmt1 + stmt2


    def p_if(self, p):
        '''if : IF C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE
        '''
        cond, stmt = p[3], p[6]
        fn = {"type" : f"if {cond}" , "children" : stmt}
        p[0] = fn

    def p_ifelse(self, p):
        '''ifelse : IFELSE C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE ELSE E_LBRACE stmt E_RBRACE
        '''
        cond, stmt1, stmt2 = p[3], p[6], p[10]
        p[0] = {
            "type" : f"ifelse {cond}", "children" :[ 
                {"type" : "do", "children" : stmt1},
                {"type" : "else", "children" : stmt2}]
            }

    def p_while(self, p):
        '''while : WHILE C_LBRACE NO_MARKER C_RBRACE W_LBRACE stmt W_RBRACE
        '''
        cond, stmt = p[3], p[6]
        p[0] = {"type" : "while" , "children" : stmt}


    def p_repeat(self, p):
        '''repeat : REPEAT cste R_LBRACE stmt R_RBRACE
        '''
        cste, stmt = p[2], p[4]
        p[0] = {"type" : f"repeat {cste}", "children" : stmt}


    def p_cond(self, p):
        '''cond :  PATH_RIGHT
                 | PATH_AHEAD 
                 | PATH_LEFT
        '''
        p[0] = p[1]

             
    def p_action(self, p):
        '''action : MOVE 
                  | TURN_RIGHT
                  | TURN_LEFT

        '''
        p[0] = {"type" : p[1]}

    def p_cste(self, p):
        '''cste : INT
        '''
        p[0] = p[1]
        

    def p_error(self, p):
        raise SyntaxError
    
    def parse(self, code) : 
        return self.yacc.parse(code)
