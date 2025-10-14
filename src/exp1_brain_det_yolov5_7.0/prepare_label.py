import os 
import xml.etree.ElementTree as ET
import pathlib
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    box_dict = {}
    abnormal_image_paths = []
    brain_image_paths = []
    classes = ['brain', 'abnormal']
    for rdir, _, files in os.walk('../../dataset/brain_det/images'):
        for file in files:
            image_path = os.path.join(rdir, file)
            image_file = image_path.split('/')[-1]
            file_name, file_ext = os.path.splitext(image_file)

            ann_path = image_path.replace('images', 'annotations').replace('.png', '.xml')
            if os.path.isfile(ann_path) == False:
                continue
            
            tree=ET.parse(open(ann_path))
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            boxes = []
            for obj in root.iter('object'):
                xmlbox = obj.find('bndbox')
                clsname = obj.find('name').text
                x1, x2, y1, y2 = int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text)
                x1 = min(max(0, x1), width)
                x2 = min(max(0, x2), width)
                y1 = min(max(0, y1), height)
                y2 = min(max(0, y2), height)
                if x1 >= x2 or y1 >= y2:
                    continue
                assert clsname in classes
                boxes.append([x1, y1, x2, y2, clsname])
            assert len(boxes) == 1
            x1, y1, x2, y2, clsname = boxes[0]
            if clsname == 'brain':
                brain_image_paths.append(image_path)
            else:
                abnormal_image_paths.append(image_path)
            box_dict[image_path] = [x1, y1, x2, y2, height, width, clsname]
    
    brain_train, brain_val = train_test_split(brain_image_paths, test_size=200, random_state=42)
    abnormal_train, abnormal_val = train_test_split(abnormal_image_paths, test_size=15, random_state=42)
    train_image_paths = brain_train + abnormal_train
    val_image_paths = brain_val + abnormal_val

    os.makedirs('../../dataset/brain_det/labels', exist_ok=True)
    train_file = open('../../dataset/brain_det/train.txt', 'w')
    for image_path in train_image_paths:
        x1, y1, x2, y2, height, width, clsname = box_dict[image_path]

        label_path = image_path.replace('images', 'labels').replace('.png', '.txt')
    
        label_file = open(label_path, 'w')
        xc = 0.5*(x1+x2)/width
        yc = 0.5*(y1+y2)/height
        w = (x2-x1)/width
        h = (y2-y1)/height
        label_file.write('{} {} {} {} {}\n'.format(classes.index(clsname), xc, yc, w, h))
        label_file.close()

        train_file.write(image_path + "\n")
    train_file.close()

    val_file = open('../../dataset/brain_det/val.txt', 'w')
    for image_path in val_image_paths:
        x1, y1, x2, y2, height, width, clsname = box_dict[image_path]

        label_path = image_path.replace('images', 'labels').replace('.png', '.txt')
    
        label_file = open(label_path, 'w')
        xc = 0.5*(x1+x2)/width
        yc = 0.5*(y1+y2)/height
        w = (x2-x1)/width
        h = (y2-y1)/height
        label_file.write('{} {} {} {} {}\n'.format(classes.index(clsname),xc, yc, w, h))
        label_file.close()

        val_file.write(image_path + "\n")
    val_file.close()

    dataset_dir = pathlib.Path('../../dataset/brain_det').resolve()
    data_file = open('data/brain_det.yaml', 'w')
    data_file.write('''
path: {}
train: train.txt
val: val.txt

# Classes
names:
    0: brain
    1: abnormal
'''.format(dataset_dir))
