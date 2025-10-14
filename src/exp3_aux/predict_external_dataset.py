import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from models import AneurysmAuxModel
from dataset import AneurysmTestDataset, classes
from utils import seed_everything

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default='configs/mit_b4_fpn_384.yaml', type=str)
parser.add_argument("--ckpt_dir", default='checkpoints', type=str)
parser.add_argument("--pred_dir", default='predictions', type=str)
parser.add_argument("--folds", default=[0,1,2,3,4], nargs="+", type=int)
args = parser.parse_args()
print(args)

seed_everything(seed=123)

if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)

    models = {}
    for fold in args.folds:
        model = AneurysmAuxModel(
            encoder_name=cfg['encoder_name'], 
            decoder_name=cfg['decoder_name'],
            encoder_weights=None, 
            encoder_feat_dims=cfg['encoder_feat_dims'], 
            num_classes=len(classes),
            test_mode=True
        )
        model.cuda()
        CHECKPOINT = '{}/{}_{}_fold{}.pt'.format(args.ckpt_dir, cfg['encoder_name'], cfg['image_size'], fold)
        model.load_state_dict(torch.load(CHECKPOINT))
        model.eval()
        models[fold] = model

    os.makedirs(args.pred_dir, exist_ok=True)

    for ext_data in ['Lausanne_TOFMRA', 'Royal_Brisbane_TOFMRA']:
        df = pd.read_csv('../../dataset/external/{}/data.csv'.format(ext_data))
        brain_box_df = pd.read_csv('../../dataset/external/{}/brain_box.csv'.format(ext_data))

        test_dataset = AneurysmTestDataset(
            df=df,
            brain_box_df=brain_box_df,
            image_size=cfg['image_size'])
        test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], num_workers=cfg['workers'], shuffle=False)
        print('{} | Test size: {}'.format(ext_data, len(test_loader.dataset)))

        pred_dict = {}
        for images, batch_image_paths in tqdm(test_loader):
            images = images.cuda()
            batch_image_paths = np.array(batch_image_paths)
            batch_preds = []
            with autocast("cuda"), torch.no_grad():
                for fold in args.folds:
                    outputs = torch.sigmoid(models[fold](images))
                    outputs = outputs.data.cpu().numpy()
                    batch_preds.append(outputs)
            batch_preds = np.array(batch_preds).mean(0)
            
            for image_path, pred in zip(batch_image_paths, batch_preds):
                pred_dict[image_path] = pred 
        torch.save(pred_dict, '{}/{}_{}_{}.pt'.format(args.pred_dir, cfg['encoder_name'], cfg['image_size'], ext_data))