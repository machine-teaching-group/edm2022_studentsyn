'''
code based on https://github.com/carpedm20/karel, 
and https://github.com/alts/karel
'''

from ..baseclasses.parser_base import _Parser, dummy


class _ParserKarelUnified(_Parser):
    # no __init__ method is declared, so base class _Parser's __init__ method will be used

    # class attributes
    tokens = [
            'DEF', 'RUN', 
            'M_LBRACE', 'M_RBRACE', 'C_LBRACE', 'C_RBRACE', 'R_LBRACE', 'R_RBRACE',
            'W_LBRACE', 'W_RBRACE', 'I_LBRACE', 'I_RBRACE', 'E_LBRACE', 'E_RBRACE',
            'INT', #'NEWLINE', 'SEMI', 
            'WHILE', 'REPEAT',
            'IF', 'IFELSE', 'ELSE',
            'FRONT_IS_CLEAR', 'NO_FRONT_IS_CLEAR', 'LEFT_IS_CLEAR', 'NO_LEFT_IS_CLEAR',
            'RIGHT_IS_CLEAR', 'NO_RIGHT_IS_CLEAR', 'MARKERS_PRESENT', 'NO_MARKERS_PRESENT',
            'MOVE', 'TURN_RIGHT', 'TURN_LEFT',
            'PICK_MARKER', 'PUT_MARKER',
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

    t_DEF = 'DEF'
    t_RUN = 'run'
    t_WHILE = 'WHILE'
    t_REPEAT = 'REPEAT'
    t_IF = 'IF'
    t_IFELSE = 'IFELSE'
    t_ELSE = 'ELSE'

    t_FRONT_IS_CLEAR = 'bool_path_ahead'
    t_NO_FRONT_IS_CLEAR = 'bool_no_path_ahead'
    t_LEFT_IS_CLEAR = 'bool_path_left'
    t_NO_LEFT_IS_CLEAR = 'bool_no_path_left'
    t_RIGHT_IS_CLEAR = 'bool_path_right'
    t_NO_RIGHT_IS_CLEAR = 'bool_no_path_right'
    t_MARKERS_PRESENT = 'bool_marker'
    t_NO_MARKERS_PRESENT = 'bool_no_marker'

    conditional_functions = [
            t_FRONT_IS_CLEAR, t_NO_FRONT_IS_CLEAR,
            t_LEFT_IS_CLEAR, t_NO_LEFT_IS_CLEAR,
            t_RIGHT_IS_CLEAR, t_NO_RIGHT_IS_CLEAR,
            t_MARKERS_PRESENT, t_NO_MARKERS_PRESENT
    ]

    t_MOVE = 'move'
    t_TURN_RIGHT = 'turn_right'
    t_TURN_LEFT = 'turn_left'
    t_PICK_MARKER = 'pick_marker'
    t_PUT_MARKER = 'put_marker'

    action_functions = [
            t_MOVE,
            t_TURN_RIGHT, t_TURN_LEFT,
            t_PICK_MARKER, t_PUT_MARKER,
    ]

    #########
    # lexer
    #########

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
        print("Illegal character %s" % repr(t.value[0]))
        t.lexer.skip(1)


    #########
    # parser
    #########


    def p_prog(self, p):
        '''prog : DEF RUN M_LBRACE stmt M_RBRACE'''
        stmt = p[4]

        def fn():
            return stmt()
        p[0] = fn


    def p_stmt(self, p):
        '''stmt : while
                | repeat
                | stmt_stmt
                | action
                | if
                | ifelse
        '''
        function = p[1]

        def fn():
            return function()
        p[0] = fn


    def p_stmt_stmt(self, p):
        '''stmt_stmt : stmt stmt
        '''
        stmt1, stmt2 = p[1], p[2]

        def fn():
            stmt1(); stmt2();
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
                    out = dummy()
                return out
        else:
            def fn():
                if cond():
                    out = stmt()
                else:
                    out = dummy()
                return out

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
        else:
            def fn():
                if cond():
                    out = stmt1()
                else:
                    out = stmt2()
                return out

        p[0] = fn


    def p_while(self, p):
        '''while : WHILE C_LBRACE cond C_RBRACE W_LBRACE stmt W_RBRACE
        '''
        cond, stmt = p[3], p[6]

        if self.hit_info is not None:
            true_key = "while_true" + str(self.counter)
            self.hit_info[true_key] = 0
            self.counter += 1  

            def fn():
                while(cond()):
                    self.hit_info[true_key] += 1
                    self.symbolic_decisions.append("w1")
                    stmt()
                else:
                    self.symbolic_decisions.append("w0")
                    dummy()

        else:
            def fn():
                while(cond()):
                    stmt()
        p[0] = fn


    def p_repeat(self, p):
        '''repeat : REPEAT cste R_LBRACE stmt R_RBRACE
        '''
        cste, stmt = p[2], p[4]


        def fn():
            for _ in range(cste()):
                stmt()
        p[0] = fn


    def p_cond(self, p):
        '''cond : FRONT_IS_CLEAR
                | NO_FRONT_IS_CLEAR
                | LEFT_IS_CLEAR
                | NO_LEFT_IS_CLEAR
                | RIGHT_IS_CLEAR
                | NO_RIGHT_IS_CLEAR
                | MARKERS_PRESENT
                | NO_MARKERS_PRESENT
        '''
        cond = p[1]
        def fn():
            return getattr(self.env, cond)()

        p[0] = fn


    def p_action(self, p):
        '''action : MOVE
                  | TURN_RIGHT
                  | TURN_LEFT
                  | PICK_MARKER
                  | PUT_MARKER
        '''
        action = p[1]
        def fn():
            return getattr(self.env, action)()
        p[0] = fn


    def p_cste(self, p):
        '''cste : INT
        '''
        value = p[1]
        p[0] = lambda: int(value)


    def p_error(self, p):
        if p:
            print("Syntax error at '%s'" % p.value)
        else:
            print("Syntax error at EOF")
            