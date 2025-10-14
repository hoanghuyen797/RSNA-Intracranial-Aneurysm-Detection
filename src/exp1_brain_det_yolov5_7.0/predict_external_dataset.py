import os 
import cv2 
import numpy as np 
import pandas as pd
import torch 
from tqdm import tqdm 
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

def img2tensor_yolov5(img, imgsz, stride, device):
    img_tensor = letterbox(img, imgsz, stride=stride, auto=True)[0]
    img_tensor = img_tensor.transpose((2, 0, 1))
    img_tensor = np.ascontiguousarray(img_tensor)

    img_tensor = torch.from_numpy(img_tensor).to(device)
    img_tensor = img_tensor.half()
    img_tensor /= 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

if __name__ == "__main__":
    brain_det_model = DetectMultiBackend('checkpoints/yolov5n_640/weights/brain_det_yolov5n_640.pt', device=select_device(0), dnn=False, fp16=True)
    brain_det_model.eval()
    brain_det_model.warmup(imgsz=(1, 3, 640, 640))

    for ext_data in ['Lausanne_TOFMRA', 'Royal_Brisbane_TOFMRA']:
        df_data = []
        for rdir, _, files in os.walk('../../dataset/external/{}/brain_det'.format(ext_data)):
            for file in tqdm(files):
                image_path = os.path.join(rdir, file)
                file_name, file_ext = os.path.splitext(file)
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                brain_height, brain_width = image.shape[0:2]
                image = np.stack([image, image, image], -1)
                brain_image = img2tensor_yolov5(image, imgsz=640, stride=32, device=select_device(0))

                boxes = []
                labels = []
                scores = []
                with torch.no_grad():
                    brain_dets = brain_det_model(brain_image, augment=False, visualize=False)
                    brain_dets = non_max_suppression(brain_dets, conf_thres=0.1, iou_thres=0.4, classes=None, agnostic=True, max_det=100)

                    for det in brain_dets:
                        if len(det):
                            det[:, :4] = scale_boxes((640, 640), det[:, :4], (brain_height, brain_width)).round()
                            det = det.data.cpu().numpy()
                            for d in det:
                                x1, y1, x2, y2, score, label = int(d[0]), int(d[1]), int(d[2]), int(d[3]), float(d[4]), int(d[5])
                                boxes.append([x1, y1, x2, y2])
                                scores.append(score)
                                labels.append(label)
                if len(boxes) > 0:
                    boxes = np.array(boxes)
                    scores = np.array(scores)
                    labels = np.array(labels)
                    
                    idxs = np.argsort(scores)[::-1]
                    scores = scores[idxs]
                    boxes = boxes[idxs,:]
                    labels = labels[idxs]
                    brain_x1, brain_y1, brain_x2, brain_y2 = boxes[0,:]
                    if labels[0] == 0:
                        brain_class_name = 'brain'
                    else:
                        brain_class_name = 'abnormal'
                else:
                    brain_x1, brain_y1, brain_x2, brain_y2, brain_class_name = 0, 0, brain_height, brain_width, 'brain'

                df_data.append([file_name, '{} {} {} {} {}'.format(brain_x1, brain_y1, brain_x2, brain_y2, brain_class_name)])
        new_df = pd.DataFrame(data=np.array(df_data), columns=['SeriesInstanceUID', 'brain_box'])
        new_df.to_csv('../../dataset/external/{}/brain_box.csv'.format(ext_data), index=False)
