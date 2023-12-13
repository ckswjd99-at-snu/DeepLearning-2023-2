import os
if not os.path.exists('../ParlAI'):
    os.system('git clone https://github.com/facebookresearch/ParlAI.git ../ParlAI')
import sys
sys.path.append('../ParlAI')
import torch
import logging
import argparse
import pandas as pd
# from utils.eval_utils import Evaluation
from utils.eval_utils import Evaluation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--response-path', type=str, help='Input your own response path')
    parser.add_argument('--eval-mode', type=str, help='Select evaluation protocol', default='safety_score')
    args = parser.parse_args()
    
    return args


def main(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load generated response
    results = pd.read_csv(args.response_path, delimiter='|')
    contexts = results['context'].tolist()
    responses = results['response'].tolist()
    
    # Make Evaluation class
    evaluation = Evaluation(args.eval_mode, device=device)
    
    # Run evaluation
    safety_score = evaluation.eval_safety_cls(contexts, responses)

    print(f"Safety score: {safety_score:.4f}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
