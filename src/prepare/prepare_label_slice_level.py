import os 
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET

if __name__ == "__main__":
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
    slices_df = pd.read_csv('../../dataset/train_slice_level.csv')
    slices_df = slices_df[['SeriesInstanceUID','SOPInstanceUID','z','image_path']]
    kfold_df = pd.read_csv('../../dataset/train_kfold.csv')
    kfold_dict = dict(zip(kfold_df.SeriesInstanceUID.values.tolist(), kfold_df.fold.values.tolist()))

    df = pd.read_csv('../../dataset/train.csv')
    columns = df.columns.values.tolist()
    columns.remove('SeriesInstanceUID')
    # print('*'*20)
    # print(columns)
    label_dict = {}
    for _, row in df.iterrows():
        label_dict[row['SeriesInstanceUID']] = row[columns].values.tolist()
    # print('*'*20)

    ann_data = []
    meta = []
    has_box = []
    for _, row in tqdm(slices_df.iterrows(), total=len(slices_df)):
        image_path = row['image_path']
        ann_path = image_path.replace('images','annotations').replace('.png', '.xml')
        ann = ''
        if os.path.isfile(ann_path):
            tree=ET.parse(open(ann_path))
            root = tree.getroot()
            ann = []
            for obj in root.iter('object'):
                clsname = obj.find('name').text
                xmlbox = obj.find('bndbox')
                x1, x2, y1, y2 = int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text)
                ann.append('{} {} {} {} {}'.format(clsname, x1, y1, x2, y2))
            if len(ann) > 0:
                has_box.append(1)
            else:
                has_box.append(0)
            ann = '|'.join(ann)
        else:
            has_box.append(0)
        meta.append(label_dict[row['SeriesInstanceUID']] + [kfold_dict[row['SeriesInstanceUID']]])
        ann_data.append(ann)
    slices_df[columns + ['fold']] = np.array(meta)
    slices_df['annotation'] = np.array(ann_data, dtype=str)
    slices_df['has_box'] = np.array(has_box, dtype=int)
    slices_df[classes] = slices_df[classes].astype(int)
    slices_df['fold'] = slices_df['fold'].astype(int)
    print(slices_df.shape)
    print(slices_df.head())
    slices_df.to_csv('../../dataset/train_slice_level_kfold.csv', index=False)

    pos_df = slices_df.loc[(slices_df['Aneurysm Present'] == 1)*(slices_df['has_box']==1)]
    print(pos_df.shape)


    brain_det_classes = ['brain', 'abnormal']
    brain_det_df_data = []
    for rdir, _, files in os.walk('../../dataset/brain_det/images'):
        for file in files:
            image_path = os.path.join(rdir, file)
            image_file = image_path.split('/')[-1]
            file_name, file_ext = os.path.splitext(image_file)

            ann_path = image_path.replace('images', 'annotations').replace('.png', '.xml')
            if os.path.isfile(ann_path) == True:
                tree=ET.parse(open(ann_path))
                root = tree.getroot()
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)

                boxes = []
                for obj in root.iter('object'):
                    xmlbox = obj.find('bndbox')
                    clsname = obj.find('name').text
                    assert clsname in brain_det_classes
                    x1, x2, y1, y2 = int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text)
                    x1 = min(max(0, x1), width)
                    x2 = min(max(0, x2), width)
                    y1 = min(max(0, y1), height)
                    y2 = min(max(0, y2), height)
                    if x1 >= x2 or y1 >= y2:
                        continue
                    boxes.append([x1, y1, x2, y2, clsname])
                assert len(boxes) == 1
                x1, y1, x2, y2, clsname = boxes[0]
            else:
                image = cv2.imread(image_path)
                height, width = image.shape[0:2]
                x1, y1, x2, y2, clsname = 0, 0, width, height, 'brain'
            brain_det_df_data.append([file_name, '{} {} {} {} {}'.format(x1, y1, x2, y2, clsname)])
    brain_det_df = pd.DataFrame(data=brain_det_df_data, columns=['SeriesInstanceUID', 'brain_box'])
    brain_det_df.to_csv('../../dataset/brain_box.csv', index=False)
    print(brain_det_df.shape)
    print(brain_det_df.head())