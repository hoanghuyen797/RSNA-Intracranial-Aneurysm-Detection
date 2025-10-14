import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore")

classes = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present'
]

if __name__ == "__main__":
    df = pd.read_csv('../../dataset/train_slice_level_kfold.csv')
   
    vit_large_384_pred_dict = {}
    mit_b4_384_pred_dict = {}
    for fold in range(5):
        vit_large_384_pred_dict.update(torch.load('../exp2_cls/predictions/vit_large_patch14_clip_336.openai_ft_in12k_in1k_384_fold{}.pt'.format(fold), weights_only=False))
        mit_b4_384_pred_dict.update(torch.load('../exp3_aux/predictions/mit_b4_384_fold{}.pt'.format(fold), weights_only=False))
    
    pseudo_data = []
    for fold in range(5):
        det_model = YOLO("../exp0_aneurysm_det/checkpoints/yolo11x_1280_fold{}/weights/best.pt".format(fold))
        print('*'*30, 'Fold {}'.format(fold), '*'*30)
        val_df = df.loc[df['fold'] == fold]

        for SeriesInstanceUID in tqdm(np.unique(val_df.SeriesInstanceUID.values)):
            grp = val_df.loc[val_df['SeriesInstanceUID'] == SeriesInstanceUID]
            grp = grp.sort_values(by='z', ascending=True).reset_index(drop=True)

            ser_gt = []
            for class_name in classes:
                assert len(np.unique(grp[class_name].values)) == 1
                ser_gt.append(grp[class_name].values[0])
            ser_gt = np.array(ser_gt)

            ser_pred = []
            for idx, row in grp.iterrows():
                pred1 = mit_b4_384_pred_dict[row['image_path']]
                pred2 = vit_large_384_pred_dict[row['image_path']]
                pred = 0.5*pred1 + 0.5*pred2
                ser_pred.append(pred)
            ser_pred = np.array(ser_pred)
            ser_pred_max = np.max(ser_pred, 0)
            
            if np.max(ser_gt) == 0 and np.max(ser_pred_max[:-1]) > 0.9 and ser_pred_max[-1] > 0.9:
                pseudo_label = list(ser_pred_max)
                for idx, row in grp.iterrows():
                    image_path = row['image_path']
                    pred1 = mit_b4_384_pred_dict[image_path]
                    pred2 = vit_large_384_pred_dict[image_path]
                    pred = 0.5*pred1 + 0.5*pred2

                    aneurysm_idxs = np.where(pred > 0.9)[0]

                    if np.max(pred[:-1]) > 0.9 and pred[-1] > 0.9:
                        
                        det = det_model(image_path, imgsz=1280, conf=0.2, iou=0.4, augment=False, device="cuda:0", verbose=False)[0]
                        boxes = det.boxes.xyxy.data.cpu().numpy().astype(int)
                        scores = det.boxes.conf.data.cpu().numpy()
                        labels = det.boxes.cls.data.cpu().numpy().astype(int)
                        if boxes.shape[0] > 0:
                            det_idxs = np.argsort(scores)[::-1]
                            scores = scores[det_idxs]
                            boxes = boxes[det_idxs,:]
                            labels = labels[det_idxs]
                            boxes = boxes[0:len(aneurysm_idxs), :]
                            labels = labels[0:len(aneurysm_idxs)]
                            ann = []
                            for box, label in zip(boxes, labels):
                                x1, y1, x2, y2 = box
                                if label == 0:
                                    clsname = 'aneurysm'
                                else:
                                    clsname = 'aneurysm_mri_t2'
                                ann.append('{} {} {} {} {}'.format(clsname, x1, y1, x2, y2))
                            if len(ann) > 0:
                                ann = '|'.join(ann)
                                pseudo_data.append([SeriesInstanceUID, row['z'], row['image_path']] + pseudo_label + [ann, 1, fold])
    new_df = pd.DataFrame(data=np.array(pseudo_data), columns=['SeriesInstanceUID', 'z', 'image_path'] + classes + ['annotation', 'has_box', 'fold'])
    new_df[['z'] + classes] = new_df[['z'] + classes].astype(float)
    new_df[['has_box', 'fold']] = new_df[['has_box', 'fold']].astype(int)
    new_df.to_csv('../../dataset/neg_noise_exp23.csv', index=False)
