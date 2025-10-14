import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import yaml
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from models import AneurysmAuxModel
from dataset import AneurysmDataset, classes
from utils import seed_everything, get_weighted_auc

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default='configs/mit_b4_fpn_384.yaml', type=str)
parser.add_argument("--ckpt_dir", default='checkpoints', type=str)
parser.add_argument("--folds", default=[0,1,2,3,4], nargs="+", type=int)
parser.add_argument("--pred_dir", default='predictions', type=str)
parser.add_argument("--crop_ratio", default=1, type=float)
parser.add_argument("--hflip", default=False, type=lambda x: (str(x).lower() == "true"))
args = parser.parse_args()
print(args)

seed_everything(seed=123)

if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)
    
    df = pd.read_csv('../../dataset/train_slice_level_kfold.csv')
    brain_box_df = pd.read_csv('../../dataset/brain_box.csv')

    brain_box_dict = {}
    for _, row in brain_box_df.iterrows():
        box = row['brain_box'].split(' ')
        x1, y1, x2, y2, class_name = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]
        brain_box_dict[row['SeriesInstanceUID']] = [x1, y1, x2, y2, class_name]

    os.makedirs(args.pred_dir, exist_ok=True)
    oof_gts = []
    oof_preds = []
    for fold in args.folds:
        print('*'*30, 'Fold {}'.format(fold), '*'*30)
        
        val_df = df.loc[df['fold'] == fold]
        brain_box_val_df = brain_box_df.loc[brain_box_df.SeriesInstanceUID.isin(list(np.unique(val_df.SeriesInstanceUID.values)))]
        
        if args.crop_ratio == 1:
            if args.hflip:
                pred_path = '{}/{}_{}_fold{}_hflip.pt'.format(args.pred_dir, cfg['encoder_name'], cfg['image_size'], fold)
            else:
                pred_path = '{}/{}_{}_fold{}.pt'.format(args.pred_dir, cfg['encoder_name'], cfg['image_size'], fold)
        else:
            if args.hflip:
                pred_path = '{}/{}_{}_fold{}_crop{:.2f}_hflip.pt'.format(args.pred_dir, cfg['encoder_name'], cfg['image_size'], fold, args.crop_ratio)
            else:
                pred_path = '{}/{}_{}_fold{}_crop{:.2f}.pt'.format(args.pred_dir, cfg['encoder_name'], cfg['image_size'], fold, args.crop_ratio)
        if os.path.isfile(pred_path):
            pred_dict = torch.load(pred_path, weights_only=False)
        else:
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

            val_dataset = AneurysmDataset(
                df=val_df,
                brain_box_df=brain_box_val_df,
                image_size=cfg['image_size'],
                mode='valid',
                crop_ratio=args.crop_ratio)
            val_loader = DataLoader(val_dataset, batch_size=8, num_workers=cfg['workers'], shuffle=False)
            print('VALID: {}'.format(len(val_loader.dataset)))

            pred_dict = {}
            for images, labels, batch_image_paths in tqdm(val_loader):
                images = images.cuda()
                labels = labels.cuda()
                batch_image_paths = np.array(batch_image_paths)
                with autocast("cuda"), torch.no_grad():
                    if args.hflip:
                        outputs = torch.sigmoid(model(torch.flip(images, dims=(3,))))
                        outputs = outputs.data.cpu().numpy()
                        outputs = outputs[:,[1,0,3,2,5,4,6,8,7,10,9,11,12,13]]
                    else:
                        outputs = torch.sigmoid(model(images))
                        outputs = outputs.data.cpu().numpy()

                for image_path, pred in zip(batch_image_paths, outputs):
                    pred_dict[image_path] = pred 
            torch.save(pred_dict, pred_path)
        
        print('Evaluating...')
        gts = []
        preds = []
        for SeriesInstanceUID in tqdm(np.unique(val_df.SeriesInstanceUID.values)):
            grp = val_df.loc[val_df['SeriesInstanceUID'] == SeriesInstanceUID]
            grp = grp.sort_values(by='z', ascending=True).reset_index(drop=True)

            assert len(np.unique(grp.Modality.values)) == 1
            Modality = grp.Modality.values[0]
            brain_x1, brain_y1, brain_x2, brain_y2, brain_class_name = brain_box_dict[SeriesInstanceUID]

            ser_gt = []
            for class_name in classes:
                assert len(np.unique(grp[class_name].values)) == 1
                ser_gt.append(grp[class_name].values[0])
            ser_gt = np.array(ser_gt)
            ser_pred = []
            for _, row in grp.iterrows():
                pred = pred_dict[row['image_path']]

                ser_pred.append(pred)
            ser_pred = np.array(ser_pred)
            ser_pred = np.max(ser_pred, 0)
            
            gts.append(ser_gt)
            preds.append(ser_pred)
        
        oof_gts.extend(gts)
        oof_preds.extend(preds)

        gts = np.array(gts)
        preds = np.array(preds)
        auc_score = get_weighted_auc(gts, preds)
        print('Fold {} | auc_score {:.4f}'.format(fold, auc_score))
    oof_gts = np.array(oof_gts)
    oof_preds = np.array(oof_preds)
    auc_score = get_weighted_auc(oof_gts, oof_preds)
    print('OOF | auc_score {:.4f}'.format(auc_score))
