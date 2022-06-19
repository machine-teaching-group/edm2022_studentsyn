import torch
import torch.nn as nn  
import torch.nn.functional as F

from code.utils.utils import * 
from .syntax.syntax import SyntaxChecker


class MapModule(nn.Module):
    '''
        MapModule abstracts the pattern of applying a module M 
        to only features of the input given that BS may be in product form 
        BS = B1 x B2 x .. 
        nb_mod_dim = how many feats from the end should the module use 
                     rest is treaded as BS 
        elt_module = the module 

    '''
    def __init__(self, elt_module, nb_mod_dim):
        super(MapModule, self).__init__()
        self.elt_module = elt_module
        self.nb_mod_dim = nb_mod_dim

    def forward(self, x):
        x_batch_shape = x.size()[:-self.nb_mod_dim]
        x_feat_shape = x.size()[-self.nb_mod_dim:]
        flat_x_shape = (-1, ) + x_feat_shape
        flat_x = x.contiguous().view(flat_x_shape)
        flat_y = self.elt_module(flat_x)

        y_feat_shape = flat_y.size()[1:]
        y_shape = x_batch_shape + y_feat_shape
        y = flat_y.view(y_shape)
        return y 



class PolicyNet(nn.Module):

    def __init__(self, vocab_size,vocab, lstm_input_size, 
        lstm_hidden_size, embedding_dim, nb_layers, check_syntax_mode = None,
        ):

        super(PolicyNet, self).__init__()
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.nb_layers = nb_layers

        self.rnn = nn.LSTM(
            self.lstm_input_size,
            self.lstm_hidden_size,
            self.nb_layers,
            batch_first = True
        )

        self.vocab_size = vocab_size
        self.out2token = MapModule(nn.Linear(self.lstm_hidden_size, self.vocab_size), 1)
        
        
        self.check_syntax_mode = check_syntax_mode
        if self.check_syntax_mode != None :             
            lstm_params = (vocab_size, embedding_dim, 
                           lstm_hidden_size, nb_layers)
            self.checker = SyntaxChecker(
                self.check_syntax_mode,
                lstm_params,
                vocab["tkn2idx"]
            )
        
    def forward(self, io_emb, action_emb, input_seqs, lstm_state = None , grammar_state = None):

        state_emb = torch.cat([io_emb,action_emb],dim = -1)
        orig_shape = state_emb.shape
        state_emb = torch.flatten(state_emb, start_dim = 0, end_dim = 1)

        if lstm_state != None : 
            # fix lstm state for parallel grids, num_io 
            (hn,cn) = lstm_state

            hn = torch.flatten(hn,1,2)
            cn = torch.flatten(cn,1,2)

            lstm_state = (hn,cn)       

        # use padded sequence for faster run time 
        BS = io_emb.shape[0]
        lens = torch.bincount(torch.where(input_seqs != 0)[0], minlength = BS).cpu()

        # edge case, if we have a sample with only <pad> 
        lens[torch.where(lens == 0)] = 1
        
        num_grids = orig_shape[1]
        lens = lens.repeat(1,num_grids).flatten()
        packed = torch.nn.utils.rnn.pack_padded_sequence(state_emb, lens, batch_first=True, enforce_sorted=False)
        logits_packed, (hn,cn) = self.rnn(packed,lstm_state)   
        logits, _  = torch.nn.utils.rnn.pad_packed_sequence(logits_packed, batch_first=True)
    
        # need to add original padding 
        pad_len = orig_shape[2] - logits.shape[1]
        logits = F.pad(logits, pad = (0,0,0,pad_len,0,0),value = 0)

        # add num_io information
        unflatten_hidden = nn.Unflatten(1, orig_shape[0:2])
        unflatten = nn.Unflatten(0, orig_shape[0:2])
        logits = unflatten(logits)       
        hn = unflatten_hidden(hn)
        cn = unflatten_hidden(cn)

        pool_out, _ = logits.max(1)

        scores = self.out2token(pool_out)
        syntax_mask = None 

        # check syntax either automatically 
        # or with a mask 
        if self.check_syntax_mode != None : 

            scores, grammar_state, syntax_mask = self.checker.forward(
                input_seqs.tolist(), 
                action_emb[:,0], # choose only one of the parrallel grids 
                scores, 
                grammar_state)

        return scores, (hn,cn) , pool_out , grammar_state, syntax_mask



class ResBlock(nn.Module):
    '''
    kernel_size: width of the kernels
    in_feats: number of channels in inputs
    This is a simple Res Network, with convolutions and intermediate ReLus 
    '''
    def __init__(self, kernel_size, in_feats):

        super(ResBlock, self).__init__()
        self.feat_size = in_feats
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.conv2 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.conv3 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = out + residual
        out = self.relu(out)
        return out

class GridEncoder(nn.Module):
    def __init__(self, kernel_size, conv_stack, fc_stack, IMG_SIZE):

        super(GridEncoder, self).__init__()
        self.conv_layers = []

        # if we want to add different kernel parameters e.g. 
        # 64 -> 32 -> .. , then we stack ResBlocks with intermediate Convs for transformation
        # else just simply add a ResBlock 

        for i in range(1, len(conv_stack)):

            if conv_stack[i-1] != conv_stack[i]:
                
                block = nn.Sequential(
                    ResBlock(kernel_size, conv_stack[i-1]),
                    nn.Conv2d(conv_stack[i-1], conv_stack[i],
                              kernel_size=kernel_size, padding= (kernel_size-1)//2 ),
                    nn.ReLU(inplace=True)
                )
            else:
                block = ResBlock(kernel_size, conv_stack[i-1])

            self.conv_layers.append(block)
            self.add_module("ConvBlock-" + str(i-1), self.conv_layers[-1])

        # Calculate fc dimension
        first_dim = conv_stack[-1] * IMG_SIZE[-1] * IMG_SIZE[-2]
        fc_stack = [first_dim] + fc_stack

        # normally we have Conv -> Lin(conv_out,out_dim)
        # and optionally we can stack more FCs with intermediate ReLus 

        self.fc_layers = []

        for i in range(1, len(fc_stack)):

            self.fc_layers.append(nn.Linear(fc_stack[i-1],fc_stack[i]))
            self.add_module("FC-" + str(i-1), self.fc_layers[-1])


    def forward(self, x):
        '''
        x ::  BS x C x H x W
        '''
        batch_size = x.size(0)

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.view(batch_size, -1)

        for i in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[i](x))
        
        x = self.fc_layers[-1](x)
        return x


class StateEncoder(nn.Module):
    '''
        Encode state of (IO, action)

        conv_stack : number of channels at each conv layer
                        (includes input) [64,...]
        fc_stack : dims of fully connected layers [...]
        embedding_dim : action embedding dim 
    '''
    def __init__(self, kernel_size, conv_stack, fc_stack, 
        embedding_dim,vocab_size, IMG_SIZE):

        super(StateEncoder, self).__init__()
        in_feat =  conv_stack[0] // 2          
        
        # embedding for action tokens 
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )
        # general encoder module 
        encoder = lambda : MapModule(
            nn.Sequential(
                nn.Conv2d(IMG_SIZE[0], in_feat,
                        kernel_size=kernel_size, padding= (kernel_size -1)//2),
                nn.ReLU(inplace=True)
            ), 3)

        self.in_grid_enc = encoder()
        self.out_grid_enc = encoder()

        self.joint_enc = MapModule(
            GridEncoder(kernel_size, conv_stack, fc_stack, IMG_SIZE), 3
            )
        
    def forward(self,input_grids,output_grids,tgt_inp_seqs):      
        max_len = tgt_inp_seqs.shape[1]
        num_io = input_grids.shape[1]  
        token_emb = self.embedding(tgt_inp_seqs).unsqueeze(1).repeat(1,num_io,1,1)
        inp_emb = self.in_grid_enc(input_grids)
        out_emb = self.out_grid_enc(output_grids)

        # IO embedding :: BS x num_io x seq_len x latent
        # replicate to max_len dimensions
        io_emb = torch.cat([inp_emb, out_emb], 2)
        joint_emb = self.joint_enc(io_emb) 
        joint_emb = joint_emb.unsqueeze(2).repeat(1,1,max_len,1)

        return joint_emb, token_emb

class Synthesizer(nn.Module):

    def __init__(self,
                 kernel_size, conv_stack, fc_stack,
                 tgt_vocabulary_size,
                 tgt_embedding_dim,
                 lstm_hidden_size,
                 nb_lstm_layers,
                 check_syntax_mode,
                 vocab,
                 IMG_SIZE):

        super(Synthesizer, self).__init__()        

        self.encoder = StateEncoder(kernel_size, conv_stack, fc_stack,
            tgt_embedding_dim,tgt_vocabulary_size, IMG_SIZE)

        lstm_input_size = fc_stack[-1] + tgt_embedding_dim 
        self.check_syntax_mode = check_syntax_mode
        self.actor = PolicyNet(tgt_vocabulary_size,
                                            vocab,
                                            lstm_input_size,
                                            lstm_hidden_size,
                                            tgt_embedding_dim,
                                            nb_lstm_layers,
                                            check_syntax_mode)


    def forward(self, input_grids, output_grids, tgt_inp_sequences, lstm_state = None, grammar_state = None):
        '''
            - input : 
                      input_grids :: BS x num_io x feat_vec x gs x gs 
                      output_grids :: ...
                      tgt_input_seqs :: BS x seq_len
                      lstm_state :: nb_layers x BS x 1 x hidden_size      
                      grammar_state :: syntactic state for generated program, see syntax.py   
            - output :
                      scores :: BS x seq_len x vocab_size
                      (hn,cn) :: lstm_state
        '''
        io_emb, token_emb = self.encoder(input_grids, output_grids, tgt_inp_sequences)
        scores, (hn,cn) , _ , grammar_state, syntax_mask = self.actor(io_emb,token_emb,tgt_inp_sequences,lstm_state, grammar_state)

        return scores , (hn,cn), grammar_state, syntax_mask


