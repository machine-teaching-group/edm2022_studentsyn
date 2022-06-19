import heapq 
from code.utils.utils import * 
import numpy as np 

import torch.nn as nn
import torch
from .data import PAD_IDX, START_IDX, END_IDX

class Beam:
    def __init__(self, nb_beams, k_best, device):
        self.device = device 
        self.nb_beams = nb_beams 
        self.k_best = k_best 

        # length of current rays 
        self.length = 1 

        # tuples of (logscore, seq)
        # list for heapq usage
        self.done_seq = []

        self.done = False

        # score of beam frontier 
        self.scores = torch.empty(self.nb_beams,device = device, dtype = torch.float64).zero_()

        # (timestep,beam) -> input_idx 
        self.ts_input_for_beam = [[START_IDX]]
        self.parentBeam = []
        
        # What do to for the next timestep
        self.decoder_next_input = None
        self.parent_beam_idxs = None
        self.num_rays = 1


    def get_sampled(self):
        return heapq.nlargest(self.k_best, self.done_seq)

    def advance(self, Lprobas) :
        ''' 
            Lprobas :: (beam_size x vocab_size)
                Log probabilities of next actions for each active ray 
        '''

        # evaluate all finishing posibilies of current top_k seqs
        if self.parentBeam != [] : 
            # calculate end score, use length normalized score 
            stop_beam_lps = (self.scores + Lprobas.select(1,END_IDX)) / self.length
            
            for idx, beam_lp in enumerate(stop_beam_lps):
   
                # avoid clearly wrong solutions
                if torch.isinf(beam_lp) : continue 
                # backtrace beam 
                beam_idx = idx 
                seq = [END_IDX]
                parent_step_idx = -1
                prev_input = self.ts_input_for_beam[parent_step_idx][beam_idx]
                while prev_input != START_IDX : 
                    seq.append(prev_input)
                    beam_idx = self.parentBeam[parent_step_idx][beam_idx]
                    parent_step_idx = parent_step_idx - 1 
                    prev_input = self.ts_input_for_beam[parent_step_idx][beam_idx]
                # we do not append start_idx 
                seq.reverse()
                # best seq and logprob for this beam 
                seq_rep = (beam_lp, seq)
                
                # this path is done, save 
                if (len(self.done_seq) < self.k_best):
                    # The heap is not yet full, just append
                    heapq.heappush(self.done_seq, seq_rep)
                else:
                    # We already have the correct number of elements so we will
                    # stay at this size
                    heapq.heappushpop(self.done_seq, seq_rep)

        # check for other tokens except <end> 
        # :: 1 x Tokens
        expand_Lprobas = torch.cat([
            Lprobas[:,0:END_IDX], Lprobas[:,END_IDX+1:],
            ],1)
        # calculate new beam scores 
        if self.parentBeam != [] : 
            prev_score = self.scores.unsqueeze(1).expand_as(expand_Lprobas)
            ext_beam_score = expand_Lprobas + prev_score

        else:
            # no prev score for first time 
            ext_beam_score = expand_Lprobas[0]

        # ext_beam_score :: nb_beams x n_Tokens
        flat_score = ext_beam_score.view(-1)
        nb_cont = flat_score.size(0)

        if self.nb_beams < nb_cont : 
            normalized_flat_score = flat_score /  self.length
            bestScores, bestScoresId =\
                normalized_flat_score.topk(self.nb_beams,0, largest = True, sorted = False)
        else : 
                # if we have less beams than required, keep them all 
            bestScores = flat_score / self.length
            bestScoresId = torch.arange(0, nb_cont, 1 , device = self.device ).long()

        # ignore -inf values 
        idx_finite = torch.isfinite(bestScores)
        bestScores = bestScores[idx_finite]
        bestScoresId = bestScoresId[idx_finite]

        # our new rays 
        # account for length norm
        self.scores = bestScores * self.length 
        self.length += 1 

        # convert flattend idxs to original array idxs 
        # we don't count <end> 
        n_tokens = Lprobas.size(1) - 1 

        # the best beams selected from prev timestep
        self.parent_beam_idxs = torch.div(bestScoresId, n_tokens, rounding_mode='floor') 
        
        self.next_inputs = bestScoresId % n_tokens

        # adjust for removing <end> in expand_Lprobas 
        adjust = (self.next_inputs >= END_IDX).long()
        self.next_inputs = self.next_inputs + adjust 
 
        # for some reason keep in list format 
        self.parentBeam.append(self.parent_beam_idxs.cpu().numpy().tolist())
        self.ts_input_for_beam.append(self.next_inputs.cpu().numpy().tolist())
    
        # how many rays are currently active for this example 
        self.num_rays = len(self.next_inputs) 


        if (len(self.done_seq) == self.k_best):
            if self.done_seq[0][0]  >= self.scores.max() / (self.length - 1):
                # There is no potential for improvement in the remaining
                # beams over the ones we already collected
                self.done = True
                self.num_rays = 0

        return self.next_inputs, self.done


def beam_sample(model, max_seq_len, input_grids, output_grids, 
    beam_size, top_k):

    check_syntax = model.check_syntax_mode
    if beam_size == 1 and top_k == 1 :    
        rets_1 = greedy_sampling(model,max_seq_len,input_grids,output_grids)
        return rets_1
  

    device = next(model.parameters()).device
    device_beam = device

    # one Beam object per example 
    batch_size = input_grids.size(0)
    beams = [
        Beam(beam_size, top_k, device_beam) 
        for _ in range(batch_size)
        ]

    # first input is just start idx 
    # generally we need batch_size * beam_size inputs     
    inp_seqs = torch.empty(
        (batch_size,1), 
        dtype=torch.int64, 
        device=device).fill_(START_IDX)

    logsoft  = nn.LogSoftmax(dim=-1)     

    batch_actions = inp_seqs
    batch_state = None 
    batch_input = input_grids
    batch_output = output_grids
    batch_grammar_state = None 

    for t in range(max_seq_len):
        # :: BS x beam_size , 1 , n_tokens   
        next_actions, lstm_state, grammar_state, _ = \
            model(batch_input,batch_output,batch_actions, batch_state, batch_grammar_state)

        # :: BS x beam_size x n_tokens
        next_actions = next_actions.squeeze(1)
        lbp_out = logsoft(next_actions)

        batch_actions = []
        batch_output = []
        batch_input = []
        parent_idxs = []

        curr_idx = 0 

        # update beams, select next best actions 
        # based on probability of paths

        for i, beamState in enumerate(beams) : 

            if beamState.done : continue 

            rays = beamState.num_rays
            beam_input = torch.narrow(lbp_out,0,curr_idx,rays)
            beam_input = beam_input.to(device_beam)
            # watch out, internal values change after advance 
            next_inputs, is_done = beamState.advance(beam_input)
            next_inputs = next_inputs.to(device)
            # rays gets updated after advance, to reflect the new number of active beams 
            rays_new = beamState.num_rays

            if not is_done : 
                # append prepare next_iteration 
                parent_beam_idxs = beamState.parent_beam_idxs.to(device)
                parent_idxs.append(parent_beam_idxs + curr_idx)
                batch_actions.append(next_inputs.view(-1,1))
      
                # we re-calculate IO embedding each time
                batch_output.append(output_grids[i].unsqueeze(0).repeat(rays_new,1,1,1,1))
                batch_input.append(input_grids[i].unsqueeze(0).repeat(rays_new,1,1,1,1))

            curr_idx += rays 
        
        # All of our beams are done                    
        if batch_actions == [] : break

        # calculate model input for next timestep 
        batch_actions = torch.cat(batch_actions,0)
        batch_output = torch.cat(batch_output,0)
        batch_input = torch.cat(batch_input,0)
        parent_idxs = torch.cat(parent_idxs,0)

        batch_state = (
            lstm_state[0].index_select(1, parent_idxs),
            lstm_state[1].index_select(1, parent_idxs)
        )

        if check_syntax == 'checker':
            parent_idxs = parent_idxs.cpu().numpy()
            batch_grammar_state = np.array(grammar_state)[parent_idxs].tolist()
            for state in batch_grammar_state : 
                state[0] = state[0].__copy__()
                state[1] = state[1].__copy__()

        elif check_syntax == 'learned':

            batch_grammar_state = (
                grammar_state[0].index_select(1, parent_idxs),
                grammar_state[1].index_select(1, parent_idxs)
            )

    # collect samples from all beams
    samples = [beamState.get_sampled() for beamState in beams]

    return samples 
            


def greedy_sampling(model, max_seq_len, input_grids, output_grids):
    '''
        simple util for max sampling a model
        (top_k = 1, beam_size = 1)
    '''
    device = next(model.parameters()).device

    batch_size = input_grids.size(0)

    # <s> for start actions
    actions = torch.empty(
        (batch_size,1), 
        dtype=torch.int64, 
        device=device).fill_(START_IDX)

    samples = []

    lstm_state = None 
    grammar_state = None 

    logsoft  = nn.LogSoftmax(dim=-1) 
    log_probs = torch.zeros((batch_size)).to(device)


    done = torch.zeros(batch_size, dtype = torch.bool , device = device)
    lens = 0
    # sample model 
    for step in range(max_seq_len):
        next_actions , lstm_state, grammar_state, _ = model(input_grids,output_grids, actions, lstm_state, grammar_state)  
        # get log-prog and actions
        lbp_out = logsoft(next_actions)
        max_probs, actions = torch.max(lbp_out,-1)

        samples.append(actions)
        log_probs += max_probs.squeeze().to(device) * (~done) 
        lens += (~done)

        if (done == True).all() : break 
        done = torch.logical_or(done, actions.squeeze() == END_IDX)


    samples = torch.cat(samples,-1)
    rets = []

    # pack programs and probs in the required way 
    for dec, prob, lseq in zip(samples,log_probs, lens) : 
        prob = prob / lseq
        try : 
            end_pos = torch.where(dec == END_IDX)[0][0]
            dec = dec[:end_pos+1].cpu().tolist()
        except : 
            dec = dec.cpu().tolist()
            dec.append(END_IDX)

        rets.append([(prob,dec )])
    return rets


