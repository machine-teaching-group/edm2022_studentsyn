import torch
import os 
import logging 

from .train_embeddings import DistanceRNN
from .data import StudentData, get_embed_tensor, get_vocab_size, HIDDEN_DIM
from .train_embeddings import embed_tokens
from .utils import ast_utils

from code.utils.utils import EMB_DIR

class EmbUtil:
    '''
        class used to calculate a codes embeddings
        given embedding tokens    
    '''
    def __init__(self, task, model_dir):
        self.source_task = task  
        self.model_dir = model_dir

        self.load_emb_model()

    def load_emb_model(self):
        vocab_size = get_vocab_size[self.source_task]

        model = DistanceRNN(vocab_size, HIDDEN_DIM)
        path_to_model = EMB_DIR + f'/{self.source_task}/{self.model_dir}/{self.source_task}_best.pt'
        if not os.path.exists(path_to_model):
            raise Exception('could not find embedding model')

        checkpoint = torch.load(path_to_model, 
             map_location=torch.device('cpu'))

        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        self.model = model 
        logging.info(f'loaded embedding model from {path_to_model}')

    def get_emb(self, student_ast_list):
        node_tokens = []
        for ast_code in student_ast_list: 
            node = ast_utils.ASTNode.from_json(ast_code)
            tokens = node.tokenize()
            node_tokens.append(tokens)

        embed_tensor = get_embed_tensor[self.source_task]
        with torch.no_grad():
            embs = embed_tokens(node_tokens, self.model, embed_tensor)

        return embs 



if __name__ == '__main__' : 
    util = EmbUtil('hoc4')
    data = StudentData('hoc4')
    embedding = util.get_emb([data.ast_codes[0]])