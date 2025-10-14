import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import yaml
import random
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from models import AneurysmAuxModel
from dataset import AneurysmDataset, classes
from utils import seed_everything, get_weighted_auc
from timm.utils.model_ema import ModelEmaV2
import segmentation_models_pytorch as smp
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default='configs/mit_b4_fpn_384.yaml', type=str)
parser.add_argument("--ckpt_dir", default='checkpoints', type=str)
parser.add_argument("--frac", default=1, type=float)
parser.add_argument("--folds", default=[0,1,2,3,4], nargs="+", type=int)

args = parser.parse_args()
print(args)

seed_everything(seed=123)

if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)
    
    os.makedirs(args.ckpt_dir, exist_ok = True)

    df = pd.read_csv('../../dataset/train_slice_level_kfold.csv')
    noise_df = pd.read_csv('../../dataset/neg_noise_exp23.csv')
    brain_box_df = pd.read_csv('../../dataset/brain_box.csv')

    df1 = pd.read_csv('../../dataset/external/Lausanne_TOFMRA/pseudo_exp23.csv')
    brain_box_df1 = pd.read_csv('../../dataset/external/Lausanne_TOFMRA/brain_box.csv')

    df2 = pd.read_csv('../../dataset/external/Royal_Brisbane_TOFMRA/pseudo_exp23.csv')
    brain_box_df2 = pd.read_csv('../../dataset/external/Royal_Brisbane_TOFMRA/brain_box.csv')
    
    pseudo_df = pd.concat([df1, df2], ignore_index=True)
    pseudo_brain_box_df = pd.concat([brain_box_df1, brain_box_df2], ignore_index=True)
    pseudo_brain_box_df = pseudo_brain_box_df.loc[pseudo_brain_box_df.SeriesInstanceUID.isin(list(np.unique(pseudo_df.SeriesInstanceUID.values)))]
    print(pseudo_df.shape, pseudo_brain_box_df.shape)

    for fold in args.folds:
        print('*'*30, 'Fold {}'.format(fold), '*'*30)
        train_df = df.loc[df['fold'] != fold]
        val_df = df.loc[df['fold'] == fold]

        if args.frac != 1:
            print('Quick training...')
            train_SeriesInstanceUIDs = list(np.unique(train_df.SeriesInstanceUID.values))
            val_SeriesInstanceUIDs = list(np.unique(val_df.SeriesInstanceUID.values))
            
            train_SeriesInstanceUIDs = random.sample(train_SeriesInstanceUIDs, int(args.frac*len(train_SeriesInstanceUIDs)))
            val_SeriesInstanceUIDs = random.sample(val_SeriesInstanceUIDs, int(args.frac*len(val_SeriesInstanceUIDs)))

            train_df = train_df.loc[train_df.SeriesInstanceUID.isin(train_SeriesInstanceUIDs)]
            val_df = val_df.loc[val_df.SeriesInstanceUID.isin(val_SeriesInstanceUIDs)]

        brain_box_train_df = brain_box_df.loc[brain_box_df.SeriesInstanceUID.isin(list(np.unique(train_df.SeriesInstanceUID.values)))]
        brain_box_val_df = brain_box_df.loc[brain_box_df.SeriesInstanceUID.isin(list(np.unique(val_df.SeriesInstanceUID.values)))]
        
        train_dataset = AneurysmDataset(
            df=train_df,
            brain_box_df=brain_box_train_df,
            pseudo_df=pseudo_df, 
            pseudo_brain_box_df=pseudo_brain_box_df, 
            noise_df=noise_df,
            image_size=cfg['image_size'], 
            mode='train')
        
        val_dataset = AneurysmDataset(
            df=val_df,
            brain_box_df=brain_box_val_df,
            image_size=cfg['image_size'], 
            mode='valid')

        train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['workers'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], num_workers=cfg['workers'], shuffle=False)
        
        print('TRAIN: {} | VALID: {}'.format(len(train_loader.dataset), len(val_loader.dataset)))

        model = AneurysmAuxModel(
            encoder_name=cfg['encoder_name'], 
            decoder_name=cfg['decoder_name'],
            encoder_weights=cfg['encoder_weights'], 
            encoder_feat_dims=cfg['encoder_feat_dims'], 
            num_classes=len(classes),
            test_mode=False
        )
        model.cuda()
        cls_criterion = nn.BCEWithLogitsLoss()
        seg_criterion = smp.losses.SoftBCEWithLogitsLoss()

        model_ema = ModelEmaV2(model, decay=cfg['ema_decay'], device=torch.device("cuda"))

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['init_lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs']-1)
        
        scaler = GradScaler()

        LOG = '{}/{}_{}_fold{}.log'.format(args.ckpt_dir, cfg['encoder_name'], cfg['image_size'], fold)
        CHECKPOINT = '{}/{}_{}_fold{}.pt'.format(args.ckpt_dir, cfg['encoder_name'], cfg['image_size'], fold)
        
        logger = []
        val_auc_max = 0
        for epoch in range(cfg['epochs']):
            scheduler.step()
            model.train()
            model.test_mode = False
            model_ema.module.test_mode = False
            train_loss = []
            
            loop = tqdm(train_loader)
            for images, masks, labels in loop:
                images = images.cuda()
                masks = masks.cuda()
                labels = labels.cuda()
                
                optimizer.zero_grad()

                if cfg['mixup']:
                    if random.random() < 0.5:
                        ### mixup
                        lam = np.random.beta(0.5, 0.5)
                        rand_index = torch.randperm(images.size(0))
                        images = lam * images + (1 - lam) * images[rand_index, :,:,:]
                        masks_a, masks_b = masks, masks[rand_index,:,:]
                        labels_a, labels_b = labels, labels[rand_index]
                        
                        with autocast("cuda"):
                            outputs1, outputs2 = model(images)
                            cls_loss = lam * cls_criterion(outputs1, labels_a) + (1 - lam) * cls_criterion(outputs1, labels_b)
                            seg_loss = lam * seg_criterion(outputs2, masks_a) + (1 - lam) * seg_criterion(outputs2, masks_b)
                            loss = 0.6*cls_loss + 0.4*seg_loss
                            train_loss.append(loss.item())
                    else:
                        with autocast("cuda"):
                            outputs1, outputs2 = model(images)
                            cls_loss = cls_criterion(outputs1, labels)
                            seg_loss = seg_criterion(outputs2, masks)
                            loss = 0.6*cls_loss + 0.4*seg_loss
                            train_loss.append(loss.item())
                else:
                    with autocast("cuda"):
                        outputs1, outputs2 = model(images)
                        cls_loss = cls_criterion(outputs1, labels)
                        seg_loss = seg_criterion(outputs2, masks)
                        loss = 0.6*cls_loss + 0.4*seg_loss
                        train_loss.append(loss.item())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                model_ema.update(model)

                loop.set_description('Epoch {:02d}/{:02d} | LR: {:.5f}'.format(epoch, cfg['epochs']-1, optimizer.param_groups[0]['lr']))
                loop.set_postfix(loss=np.mean(train_loss))
            train_loss = np.mean(train_loss)

            model.test_mode = True
            model_ema.module.test_mode = True
            model.eval()
            model_ema.eval()

            pred_dict = {}
            ema_pred_dict = {}
            for images, labels, batch_image_paths in tqdm(val_loader):
                images = images.cuda()
                labels = labels.cuda()
                batch_image_paths = np.array(batch_image_paths)
                with autocast("cuda"), torch.no_grad():
                    ema_outputs = torch.sigmoid(model_ema.module(images)).data.cpu().numpy()
                
                for image_path, ema_pred in zip(batch_image_paths, ema_outputs):
                    ema_pred_dict[image_path] = ema_pred 
                
            gts = []
            ema_preds = []
            for SeriesInstanceUID in tqdm(np.unique(val_df.SeriesInstanceUID.values)):
                grp = val_df.loc[val_df['SeriesInstanceUID'] == SeriesInstanceUID]
                grp = grp.sort_values(by='z', ascending=True).reset_index(drop=True)

                ser_gt = []
                for class_name in classes:
                    assert len(np.unique(grp[class_name].values)) == 1
                    ser_gt.append(grp[class_name].values[0])
                ser_gt = np.array(ser_gt)
                
                ser_ema_pred = []
                for _, row in grp.iterrows():
                    ser_ema_pred.append(ema_pred_dict[row['image_path']])
                ser_ema_pred = np.array(ser_ema_pred)
                ser_ema_pred = np.max(ser_ema_pred, 0)
                
                gts.append(ser_gt)
                ema_preds.append(ser_ema_pred)
            
            gts = np.array(gts, dtype=np.float64)
            ema_preds = np.array(ema_preds, dtype=np.float64)
            ema_val_auc = get_weighted_auc(gts, ema_preds)
            
            print('Fold: {} | train loss: {:.5f} | ema_val_auc: {:.5f}'.format(fold, train_loss, ema_val_auc))
            logger.append([epoch, round(optimizer.param_groups[0]['lr'], 8), round(train_loss, 5), round(ema_val_auc, 5)])
            log_df = pd.DataFrame(data=np.array(logger), columns=['epoch', 'lr', 'train_loss', 'ema_val_auc'])
            log_df.to_csv(LOG, index=False)

            if val_auc_max < ema_val_auc:
                print('val auc improved from {:.5f} to {:.5f} saving model to {}'.format(val_auc_max, ema_val_auc, CHECKPOINT))
                val_auc_max = ema_val_auc
                torch.save(model_ema.module.state_dict(), CHECKPOINT)
        del model
        del model_ema