import torch
import datasets


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
            
        elif turn == 'single':
            '''
            Step 2. Build single-turn dialog
            Take only the first input of the ProsocialDialog dataset
        
            self.dialog = 

            '''
        
        
    def __len__(self):
        return len(self.dialog)
    
    
    def __getitem__(self, idx):
        return self.dialog[idx]
        

def load_dataset(dname='allenai/prosocial-dialog', turn='multi', split='test'):
    if dname == 'allenai/prosocial-dialog':
        return ProsocialDialogDataset(turn, split)
    else:
        raise NotImplementedError
