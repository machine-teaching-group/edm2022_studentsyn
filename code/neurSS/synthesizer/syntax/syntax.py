
from code.utils.utils import * 

from lark import Lark
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark import Token


import torch
import torch.nn as nn  


'''
    Code used in policy net for checking syntax 

    PySyntaxChecker:    
        At each timestep get a mask M, M_j = -inf
        if token is not allowed else 0 . Add this to weights for softmax
        This way programs become syntactically correct by construction

    SyntaxLSTM:
        Else use syntaxLSTM with token embeddings (not IO), 
        to learn syntax, add to decoder LSTM output like above. 
        Sup Loss : 
        - \sum_{i=1..N} \sum_{t=1..L} LSTMsyntax(s_t^i|s^i_{1..t-1}), 
        (don't put negative weights to valid progs) 
    
    see also: https://github.com/bunelr/GandRL_for_NPS
'''

get_grammar = {

    "hoc4" : r'''
        start : START DEF  RUN  S  action  E
        action : MOVE
            | L 
            | R 
            | action action    
        START : "<s>"
        DEF : "DEF"
        RUN : "run"
        S : "m("
        E : "m)"
        MOVE : "move"
        L : "turn_left"
        R : "turn_right"

        %ignore /\s+/    
    ''',

    "hoc18" : r'''    
        start : START DEF RUN M_LBRACE stmt M_RBRACE
        stmt :      while 
                |   stmt_stmt
                |   action
                |   ifelse

        stmt_stmt : stmt stmt 
        ifelse : IFELSE C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE ELSE E_LBRACE stmt E_RBRACE
        while : WHILE C_LBRACE NO_MARKER C_RBRACE W_LBRACE stmt W_RBRACE
        cond :  PATH_RIGHT
            |   PATH_AHEAD 
            | PATH_LEFT

        action : MOVE 
            | TURN_RIGHT
            | TURN_LEFT

        START : "<s>"
        M_LBRACE : "m("
        M_RBRACE : "m)"
        C_LBRACE : "c("
        C_RBRACE : "c)"
        W_LBRACE : "w("
        W_RBRACE : "w)"
        I_LBRACE : "i("
        I_RBRACE : "i)"
        E_LBRACE : "e("
        E_RBRACE : "e)"
        RUN : "run"
        DEF : "DEF"
        WHILE : "WHILE"
        IFELSE : "IFELSE"
        ELSE : "ELSE"
        PATH_AHEAD : "bool_path_ahead"
        PATH_LEFT : "bool_path_left"
        PATH_RIGHT : "bool_path_right"
        MOVE : "move"
        TURN_RIGHT : "turn_right"
        TURN_LEFT : "turn_left"
        NO_MARKER : "bool_no_marker"    

        %ignore /\s+/    
    '''
}


class PySyntaxChecker:
    '''
        automatically check syntax by keeping 
        also a state 
    '''
    def __init__(self, tkn2idx):
        # hacky 
        # load correct gramamr f
        if len(tkn2idx) > 10 : 
            source_task = 'hoc18'
        else : 
            source_task = 'hoc4'
        self.grammar = get_grammar[source_task]

        parser = Lark(self.grammar, parser="lalr")
        terminals = parser.terminals 

        var_2_t = {}
        for t in terminals : 
            var_2_t[t.name] = str(t.pattern).strip("'").replace('\\','')
    
        self.var_2_t = var_2_t 
        self.t_2_var = {v : k for k,v in var_2_t.items()}

        self.tkn2idx = tkn2idx
        self.idx2tkn = {v : k for k,v in tkn2idx.items()}

        self.vocab_size = len(tkn2idx)

    def get_initial_checker_state(self):
        parser = Lark(self.grammar, parser="lalr")
        inter = parser.parse_interactive()
        return [inter.parser_state, inter.lexer_state]

    def get_sequence_mask(self, state, inp_seq):
        '''
            return :: BS x vocab_size
            a mask where '1's are next invalid tokens 
            for each sequence  
        '''
        inter = Lark(self.grammar, parser="lalr").parse_interactive()
        parser = InteractiveParser(inter,state[0], state[1])

        seq_len = len(inp_seq)
        mask = torch.ones(seq_len,self.vocab_size)
        # always allow pad 
        mask[:,0] = 0
        for i,token in enumerate(inp_seq) : 
            token = self.idx2tkn[token]
            # do not advance parser
            # for pad token          
            if token != '<pad>':
                t = Token(self.t_2_var[token],token)
                parser.feed_token(t)
            
            accepts = parser.accepts()
            if '$END' not in accepts :
                accepts = [self.tkn2idx[self.var_2_t[a]] for a in accepts]
                mask[i][accepts] = 0
            else : 
                break 
        return mask,  [parser.parser_state, parser.lexer_state]

class SyntaxLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, lstm_hidden_size,
                 nb_layers):
        super(SyntaxLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.lstm_input_size = embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.nb_layers = nb_layers

        self.rnn = nn.LSTM(
            self.lstm_input_size,
            self.lstm_hidden_size,
            self.nb_layers,
            batch_first = True
        )
        # flatten here
        self.out2token = nn.Sequential(
            nn.Flatten(0,-2),
            nn.Linear(self.lstm_hidden_size, self.vocab_size),
        )

    def forward(self, inp_sequence_embedded, state):
        '''
        inp_sequence_embedded: batch_size x tgt_embedding_dim x embedding_dim
        state: 2 tuple of (nb_layers x batch_size x hidden_size)
        '''

        seq_len, batch_size, _ = inp_sequence_embedded.size()
        stx_out, state = self.rnn(inp_sequence_embedded, state)

        # unflatten here to know the size at run time
        init_shape = stx_out.shape
        unflat_util = nn.Unflatten(0,init_shape[:-1])

        stx_scores = self.out2token(stx_out)
        stx_scores = unflat_util(stx_scores)

        # batch_size x seq_len x out_vocab_size
        stx_mask = -stx_scores.exp()
        # batch_size x seq_len x out_vocab_size
        
        return stx_mask, state




class SyntaxChecker(nn.Module):
    def __init__(self,check_syntax_mode,lstm_params = None, tkn2idx = None):
        '''
            check_syntax_mode :: one of 'checker,learned'
            lstm_params = tuple (vocab_size, embedding_dim, lstm_hidden_size, nb_layers)
        '''
        super(SyntaxChecker, self).__init__()
        self.check_syntax_mode = check_syntax_mode
        if self.check_syntax_mode == 'learned' : 
            self.learned_syntax_checker = SyntaxLSTM(*lstm_params)
        else : 
            self.syntax_checker = PySyntaxChecker(tkn2idx)

    def forward(self,inp,seq_emb, decoder_logit, grammar_state):
        '''
            inp : BS x max_len
            seq_emb : seq_len x batch_size x embedding_dim, state emb without IO 
            decoder_logit : BS x num_tokens
            grammar_state : holds the syntactic information
                for learned syntax = LSTM state
                for syntax checker you pass in the same grammar state
                you don't need to get it again from checker, python modifies internally
                (by refence pass)

            syntax_mask : you return syntax mask for the syntax loss function
        '''
        syntax_mask = None
        device = decoder_logit.device
        batch_size = decoder_logit.shape[0]

        if self.check_syntax_mode == 'checker':

            if grammar_state is None:
                grammar_state = [self.syntax_checker.get_initial_checker_state()
                                    for _ in range(batch_size)]
        
            out_of_syntax_list = []
            grammar_state_new = []

            for batch_idx, inp_seq in enumerate(inp):
                
                is_valid, g_state = self.syntax_checker.get_sequence_mask(grammar_state[batch_idx],inp_seq)
                out_of_syntax_list.append(is_valid)
                grammar_state_new.append(g_state)
    
            out_of_syntax_mask = torch.stack(out_of_syntax_list, 0)
            syntax_err_pos = out_of_syntax_mask.to(device)

            syntax_mask = torch.zeros(syntax_err_pos.shape, dtype = torch.float32,device = device)
            syntax_mask.masked_fill_(syntax_err_pos > 0, -float('inf'))

            syntax_mask.requires_grad = False
            decoder_logit = decoder_logit + syntax_mask

            grammar_state = grammar_state_new

        elif self.check_syntax_mode == 'learned':

            syntax_mask, grammar_state = self.learned_syntax_checker(seq_emb, grammar_state)
            decoder_logit = decoder_logit + syntax_mask


        return decoder_logit, grammar_state, syntax_mask

