import os
import torch
import logging
import argparse
from tqdm import tqdm

from utils.data_utils import *
from utils.model_utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mname', type=str, default='facebook/blenderbot-400M-distill')
    parser.add_argument('--dname', type=str, default='allenai/prosocial-dialog')
    args = parser.parse_args()
    return args


def main(args):
    id = input(u'Input ID: ')
    
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S',
        level=logging.INFO,
    )
    
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    logging.info(f'{device} is used for response generation')
    
    os.makedirs('results', exist_ok=True)
    response_path = f'results/responses_{id}.txt'
    if os.path.isfile(response_path): os.remove(response_path)
    logging.info(f'Generated responses will be saved at {response_path}')
    
    with open(response_path, "w") as f:
        f.write("context|response\n")
    
    logging.info(f'Model loaded from {args.mname}')
    '''
    Step 1. Load model and tokenizer. You may want to use load_model function in utils/model_utils.py
    
    tokenizer = 
    model = 
    '''
    
    model.eval()
    model.to(device)
    
    logging.info(f'Data loaded from {args.dname}')
    '''
    Step 2. Load your own dataset. You may want to use load_dataset function in utils/data_utils.py
    
    dataset = 
    dataloader = 
    '''
    
    logging.info('Generating Responses...')
    '''
    Step 3. Generate responses
    for data in tqdm(dataloader):
        context = 
        reply_ids = 
        reply_txts = 
        ...    
    '''
    
    # Save contexts and responses
    with open(response_path, 'a') as f:
        for c, r in zip(context, reply_txts):
            f.write(f'{c}|{r}\n')
    
    logging.info(f'Total {len(dataset)} responses saved at {response_path}')
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
