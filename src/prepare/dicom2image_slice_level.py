import pandas as pd
import os
import numpy as np
import pydicom
import cv2
from multiprocessing import Pool
import glob

class MySeriesInstance:
    def __init__(self, SeriesInstanceUID, index, total):
        self.SeriesInstanceUID = SeriesInstanceUID
        self.index = index
        self.total = total

def dicom2image_multi_process(it):
    print('*'*10, it.index, it.total, int(100*it.index/it.total), '*'*10)
    dcm_paths = glob.glob('../../dataset/series/{}/*'.format(it.SeriesInstanceUID))
    os.makedirs('../../dataset/images/{}'.format(it.SeriesInstanceUID), exist_ok=True)
    df_data = []
    images = []
    image_paths = []
    z_pos = []
    for dcm_path in dcm_paths:
        dcm_file = pydicom.dcmread(dcm_path, force=True)

        PatientID = dcm_file.get('PatientID', None)
        StudyInstanceUID = dcm_file.get('StudyInstanceUID', None)
        assert it.SeriesInstanceUID == dcm_file.SeriesInstanceUID
        SOPInstanceUID = dcm_file.get('SOPInstanceUID', None)
        ImagePositionPatient = dcm_file.get('ImagePositionPatient', [0,0,0])
        z = ImagePositionPatient[2]
        
        Modality = dcm_file.get('Modality', 'MR')

        image = dcm_file.pixel_array.astype(np.float32)
        assert dcm_file.PhotometricInterpretation == "MONOCHROME2"

        if Modality == 'CT':
            window_center = 40
            window_width = 450
            image_min = window_center - window_width // 2
            image_max = window_center + window_width // 2
            image = np.clip(image, image_min, image_max)

        image_min = np.min(image)
        image_max = np.max(image)
        image = (image - image_min) / (image_max - image_min + 1e-7)
        image = (image * 255).astype(np.uint8)

        if image.ndim == 3:
            NumberofFrames = dcm_file.get('NumberOfFrames', None)
            assert image.shape[0] == NumberofFrames
            for i in range(NumberofFrames):
                image_i = image[i,:,:]
                image_path = '../../dataset/images/{}/{}_{}.png'.format(it.SeriesInstanceUID, SOPInstanceUID, i)
                image_paths.append(image_path)
                images.append(image_i)
                z_pos.append(i)
                df_data.append([PatientID, StudyInstanceUID, it.SeriesInstanceUID, SOPInstanceUID, i, image_path])
        else:
            image_path = '../../dataset/images/{}/{}.png'.format(it.SeriesInstanceUID, SOPInstanceUID)
            image_paths.append(image_path)
            images.append(image)
            z_pos.append(z)
            df_data.append([PatientID, StudyInstanceUID, it.SeriesInstanceUID, SOPInstanceUID, z, image_path])
            
    out_df = pd.DataFrame(data=np.array(df_data), columns=['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'z', 'image_path'])
    out_df['z'] = out_df['z'].astype(float)

    images = np.array(images)
    image_paths = np.array(image_paths)
    z_pos = np.array(z_pos)
    idxs = np.argsort(z_pos)
    images = images[idxs,:,:]
    image_paths = image_paths[idxs]

    brain_image = np.mean(images.copy(), 0)
    brain_image_min = np.min(brain_image)
    brain_image_max = np.max(brain_image)
    brain_image = (brain_image - brain_image_min) / (brain_image_max - brain_image_min + 1e-7)
    brain_image = (brain_image * 255).astype(np.uint8)

    brain_image_path = '../../dataset/brain_det/images/{}.png'.format(it.SeriesInstanceUID)
    cv2.imwrite(brain_image_path, brain_image)

    for i, image_path in enumerate(image_paths):
        if i == 0:
            pre_i = i
        else:
            pre_i = i-1
        if i == images.shape[0] - 1:
            next_i = i
        else:
            next_i = i+1
        image = images[[pre_i, i, next_i],:,:]
        image = np.transpose(image, (1,2,0))
        cv2.imwrite(image_path, image)
    
    return out_df


if __name__ == "__main__":
    df = pd.read_csv('../../dataset/train.csv')
    
    list_items = []
    for index, row in df.iterrows():
        list_items.append(MySeriesInstance(row['SeriesInstanceUID'], index, len(df)))
    os.makedirs('../../dataset/images', exist_ok=True)
    os.makedirs('../../dataset/brain_det/images', exist_ok=True)
    p = Pool(32)
    results = p.map(func=dicom2image_multi_process, iterable = list_items)
    p.close()

    new_df = pd.concat(results, ignore_index=True)
    new_df.to_csv('../../dataset/train_slice_level.csv', index=False)
