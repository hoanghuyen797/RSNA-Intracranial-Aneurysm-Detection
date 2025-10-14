import numpy as np
import pandas as pd
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

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

if __name__ == "__main__":
    det_model = YOLO("../exp0_aneurysm_det/checkpoints/yolo11x_1280_fold0/weights/best.pt")

    for ext_data in ['Lausanne_TOFMRA', 'Royal_Brisbane_TOFMRA']:
        df = pd.read_csv('../../dataset/external/{}/data.csv'.format(ext_data))
        df = df.loc[(df['aneurysm'] == 1)&(df['has_box']==1)]
        
        vit_large_384_pred_dict = torch.load('../exp2_cls/predictions/vit_large_patch14_clip_336.openai_ft_in12k_in1k_384_{}.pt'.format(ext_data), weights_only=False)
        mit_b4_384_pred_dict = torch.load('../exp3_aux/predictions/mit_b4_384_{}.pt'.format(ext_data), weights_only=False)
        
        pseudo_data = []
        for SeriesInstanceUID, grp in df.groupby('SeriesInstanceUID'):
            for idx, row in grp.iterrows():
                pred1 = vit_large_384_pred_dict[row['image_path']]
                pred2 = mit_b4_384_pred_dict[row['image_path']]
                pred = 0.5*pred1 + 0.5*pred2
                if pred[-1] > 0.5 and np.max(pred[:-1]) > 0.5:
                    gt_anns = row['annotation'].split('|')
                    gt_boxes = []
                    for gt_ann in gt_anns:
                        gt_ann = gt_ann.split(' ')
                        clsname = gt_ann[0]
                        x1, y1, x2, y2 = int(gt_ann[1]), int(gt_ann[2]), int(gt_ann[3]), int(gt_ann[4])
                        gt_boxes.append([x1, y1, x2, y2])
                    
                    det = det_model(row['image_path'], imgsz=1280, conf=0.1, iou=0.4, augment=False, device="cuda:0", verbose=False)[0]
                    boxes = det.boxes.xyxy.data.cpu().numpy().astype(int)
                    scores = det.boxes.conf.data.cpu().numpy()
                    labels = det.boxes.cls.data.cpu().numpy().astype(int)

                    if boxes.shape[0] > 0:
                        ann = []
                        for gt_box in gt_boxes:
                            max_iou = 0
                            best_box = None
                            best_label = None
                            for box, label in zip(boxes, labels):
                                iou = bb_intersection_over_union(gt_box, box)
                                if iou > max_iou:
                                    max_iou = iou 
                                    best_box = box 
                                    best_label = label
                            if max_iou > 0.1:
                                x1, y1, x2, y2 = best_box
                                if best_label == 0:
                                    clsname = 'aneurysm'
                                else:
                                    clsname = 'aneurysm_mri_t2'
                                ann.append('{} {} {} {} {}'.format(clsname, x1, y1, x2, y2))
                        if len(ann) > 0:
                            pred = list(pred)
                            pred[-1] = 1
                            ann = '|'.join(ann)
                            pseudo_data.append([SeriesInstanceUID, row['z'], row['image_path']] + pred + [ann, 1])
        new_df = pd.DataFrame(data=np.array(pseudo_data), columns=['SeriesInstanceUID', 'z', 'image_path'] + classes + ['annotation', 'has_box'])
        new_df[['z'] + classes] = new_df[['z'] + classes].astype(float)
        new_df['has_box'] = new_df['has_box'].astype(int)

        new_df.to_csv('../../dataset/external/{}/pseudo_exp23.csv'.format(ext_data), index=False)
        print(new_df.shape)