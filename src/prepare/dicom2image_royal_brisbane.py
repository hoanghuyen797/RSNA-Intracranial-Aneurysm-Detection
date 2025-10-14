import os 
import cv2 
import numpy as np 
import pandas as pd 
import nibabel as nib
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path 

if __name__ == "__main__":
    os.makedirs('../../dataset/external/Royal_Brisbane_TOFMRA/brain_det', exist_ok=True)
    df_data = []
    for x1 in Path('../../dataset/external/Royal_Brisbane_TOFMRA/derivatives').iterdir():
        if x1.is_dir():
            for x2 in Path('../../dataset/external/Royal_Brisbane_TOFMRA/derivatives/{}'.format(x1.name)).iterdir():
                if x2.is_dir():
                    SeriesInstanceUID = '{}_{}'.format(x1.name, x2.name)
                    for x3 in Path('../../dataset/external/Royal_Brisbane_TOFMRA/derivatives/{}/{}'.format(x1.name, x2.name)).iterdir():
                        if x3.is_dir():
                            mask_paths = []
                            for rdir, _, files in os.walk('../../dataset/external/Royal_Brisbane_TOFMRA/derivatives/{}/{}/{}/Nifti Aneurysm Only'.format(x1.name, x2.name, x3.name)):
                                for file in files:
                                    if 'Zone' in file:
                                        continue
                                    file_name, file_ext = os.path.splitext(file)
                                    if file_ext != '.nii':
                                        continue
                                    mask_paths.append(os.path.join(rdir,file))
                            if len(mask_paths) == 0:
                                continue
                            
                            image_paths = []
                            for rdir, _, files in os.walk('../../dataset/external/Royal_Brisbane_TOFMRA/{}/{}/{}'.format(x1.name, x2.name, x3.name)):
                                for file in files:
                                    if 'Zone' in file:
                                        continue
                                    if '.nii.gz' in file:
                                        if x1.name == 'sub-015' and x2.name == 'ses-20121216' and 'singleslab_angio' in file:
                                            continue
                                        image_paths.append(os.path.join(rdir,file))
                            if len(image_paths) == 0:
                                continue

                            assert len(image_paths) == 1
                            image_path = image_paths[0]
                            image = nib.load(image_path)
                            image = image.get_fdata()
                            image_min = np.min(image)
                            image_max = np.max(image)
                            image = (image - image_min) / (image_max - image_min + 1e-7)
                            image = (image * 255).astype(np.uint8)
                            image = np.flip(np.flip(image, axis=0), axis=1)
                            image = np.rot90(image, 3, (0,1))

                            brain_image = np.mean(image, -1)
                            brain_image_min = np.min(brain_image)
                            brain_image_max = np.max(brain_image)
                            brain_image = (brain_image - brain_image_min) / (brain_image_max - brain_image_min + 1e-7)
                            brain_image = (brain_image * 255).astype(np.uint8)

                            mask = []
                            for mask_path in mask_paths:
                                mask.append(nib.load(mask_path).get_fdata())
                            mask = np.stack(mask, -1).max(axis=-1)
                            mask = mask.astype(np.uint8)
                            mask = np.rot90(mask, 3, (0,1))

                            label = 1

                            if image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1] or image.shape[2] != mask.shape[2]:
                                # print(x1.name, x2.name, image.shape, mask.shape)
                                continue

                            cv2.imwrite('../../dataset/external/Royal_Brisbane_TOFMRA/brain_det/{}.png'.format(SeriesInstanceUID), brain_image)
                            os.makedirs('../../dataset/external/Royal_Brisbane_TOFMRA/images/{}'.format(SeriesInstanceUID), exist_ok=True)
                            
                            for i in range(image.shape[2]):
                                if i == 0:
                                    pre_i = i
                                else:
                                    pre_i = i-1
                                if i == image.shape[2] - 1:
                                    next_i = i
                                else:
                                    next_i = i+1
                                image_i = image[:,:,[pre_i, i, next_i]].copy()

                                mask_i = mask[:,:,i].copy()

                                new_image_path = '../../dataset/external/Royal_Brisbane_TOFMRA/images/{}/{}.png'.format(SeriesInstanceUID, i)
                                cv2.imwrite(new_image_path, image_i)

                                ann = []
                                if np.max(mask_i) > 0:
                                    (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(mask_i, 4, cv2.CV_32S)
                                    for i in range(1, totalLabels):
                                        b_x1 = values[i, cv2.CC_STAT_LEFT]
                                        b_y1 = values[i, cv2.CC_STAT_TOP]
                                        b_w = values[i, cv2.CC_STAT_WIDTH]
                                        b_h = values[i, cv2.CC_STAT_HEIGHT]
                                        b_x2 = b_x1+b_w
                                        b_y2 = b_y1+b_h 
                                        ann.append('aneurysm {} {} {} {}'.format(b_x1, b_y1, b_x2, b_y2))
                                if len(ann) > 0:
                                    has_box = 1
                                else:
                                    has_box = 0
                                ann = '|'.join(ann)
                                df_data.append([SeriesInstanceUID, i, new_image_path, label, ann, has_box])
    out_df = pd.DataFrame(data=np.array(df_data), columns=['SeriesInstanceUID', 'z', 'image_path', 'aneurysm', 'annotation', 'has_box'])
    out_df[['aneurysm', 'z', 'has_box']] = out_df[['aneurysm', 'z', 'has_box']].astype(int)
    out_df.to_csv('../../dataset/external/Royal_Brisbane_TOFMRA/data.csv', index=False)
    print(out_df.shape)