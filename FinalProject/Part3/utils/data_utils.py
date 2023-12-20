import torch
import datasets

def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

class ProsocialDialogDataset(torch.utils.data.Dataset):
    def __init__(self, turn, split='test'):
        """

        Args:
            turn (str): Either 'multi' or 'single'
            split (str, optional): Either 'train' or 'test'. Default to 'test'.
        """
        
        self.dataset = datasets.load_dataset("allenai/prosocial-dialog")
        self.turn = turn
        self.dialog = None # This is where you may want to save the context
        
        # Build self.dialog 
        if turn == "multi":
            '''
            Step 1. Build multi-turn dialog
            To find input-output usage of BlenderBot, you can refer to https://github.com/huggingface/transformers/blob/main/src/transformers/models/blenderbot/modeling_blenderbot.py.
            
            self.dialog = 
            
            '''

            context = self.dataset[split]['context']
            response = self.dataset[split]['response']

            if len(context) %2 != 0:
                context = context[:-1]
                response = response[:-1]

            # get two pairs of context and response, then concatenate them
            context = list_chunk(context, 2)
            response = list_chunk(response, 2)
            self.dialog = [f'{c[0]}</s>{r[0]}</s>{c[1]}' for c, r in zip(context, response)]
            
        elif turn == 'single':
            '''
            Step 2. Build single-turn dialog
            Take only the first input of the ProsocialDialog dataset
        
            self.dialog = 

            '''

            self.dialog = self.dataset[split]['context']

        
        
    def __len__(self):
        return len(self.dialog)
    
    
    def __getitem__(self, idx):
        return self.dialog[idx]
        

def load_dataset(dname='allenai/prosocial-dialog', turn='single', split='test'):
    if dname == 'allenai/prosocial-dialog':
        return ProsocialDialogDataset(turn, split)
    else:
        raise NotImplementedError