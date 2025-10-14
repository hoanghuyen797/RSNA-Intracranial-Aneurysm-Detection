import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score  
         
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_weighted_auc(ytrue, ypred):
    score1 = roc_auc_score(ytrue[:,:-1], ypred[:,:-1])
    score2 = roc_auc_score(ytrue[:,-1], ypred[:,-1])
    auc_score = 0.5*score1 + 0.5*score2
    return auc_score