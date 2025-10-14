import os 
import pandas as pd
import pathlib
import xml.etree.ElementTree as ET

if __name__ == "__main__":
    classes = ['aneurysm', 'aneurysm_mri_t2']
    df = pd.read_csv('../../dataset/train_slice_level_kfold.csv')
    pos_df = df.loc[(df['Aneurysm Present'] == 1)&(df['has_box'] == 1)]

    for SeriesInstanceUID, grp in pos_df.groupby('SeriesInstanceUID'):
        os.makedirs('../../dataset/labels/{}'.format(SeriesInstanceUID), exist_ok=True)
        for _, row in grp.iterrows():
            image_path = row['image_path']
            file = image_path.split('/')[-1]
            ann_path = image_path.replace('images', 'annotations').replace('.png', '.xml')
            if os.path.isfile(ann_path) == False or os.path.isfile(image_path) == False:
                print(image_path)
                continue
    
            label_path = image_path.replace('images', 'labels').replace('.png', '.txt')
            label_file = open(label_path, 'w')

            tree=ET.parse(open(ann_path))
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            for obj in root.iter('object'):
                xmlbox = obj.find('bndbox')
                clsname = obj.find('name').text
                x1, x2, y1, y2 = int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text)

                assert clsname in ['aneurysm', 'aneurysm_mri_t2']
                xc = 0.5*(x1+x2)/width
                yc = 0.5*(y1+y2)/height
                w = (x2-x1)/width
                h = (y2-y1)/height
                label_file.write('{} {} {} {} {}\n'.format(classes.index(clsname), xc, yc, w, h))
            label_file.close()

    os.makedirs('data', exist_ok=True)
    dataset_dir = pathlib.Path('data').resolve()
    for fold in range(5):
        val_df = pos_df.loc[pos_df['fold'] == fold]
        train_df = pos_df.loc[pos_df['fold'] != fold]

        train_file = open('data/train_fold{}.txt'.format(fold), 'w')
        for _, row in train_df.iterrows():
            image_path = row['image_path']
            if os.path.isfile(image_path) == False:
                print(image_path)
                continue
            train_file.write(image_path + "\n")
        train_file.close()

        val_file = open('data/val_fold{}.txt'.format(fold), 'w')
        for _, row in val_df.iterrows():
            image_path = row['image_path']
            if os.path.isfile(image_path) == False:
                print(image_path)
                continue
            val_file.write(image_path + "\n")
        val_file.close()

        data_file = open('data/data_fold{}.yaml'.format(fold), 'w')
        data_file.write('''
    path: {}
    train: train_fold{}.txt
    val: val_fold{}.txt

    # Classes
    names:
        0: aneurysm
        1: aneurysm_mri_t2
    '''.format(dataset_dir, fold, fold))
