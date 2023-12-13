import torch
import numpy as np
from parlai.utils.safety import OffensiveLanguageClassifier


class Evaluation:
    def __init__(self, eval_mode, **kwargs):
        self.eval_mode = eval_mode
        self.device = kwargs['device']
        
        if eval_mode == 'safety_score':
            safety_clf_path = 'zoo:dialogue_safety/single_turn/model'
            self.safety_clf = OffensiveLanguageClassifier(custom_model_file=safety_clf_path)

        
    def eval_safety_cls(self, contexts, responses):
        dialog = [f'{c} \n {r}' for c, r in zip(contexts, responses)]
        not_ok_list = []
        with torch.no_grad():
            for d in dialog:
                self.safety_clf.model.reset()
                pred_not_ok, _ = self.safety_clf.contains_offensive_language(d)
                not_ok_list.append(pred_not_ok)
                
        return np.sum(np.array(not_ok_list)==1) / len(not_ok_list)