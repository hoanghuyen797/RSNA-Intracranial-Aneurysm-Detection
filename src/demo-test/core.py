import os
import cv2
import pydicom
import timm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from ultralytics import YOLO
# yolov5
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# smp
from timm.layers import BatchNormAct2d, SelectAdaptivePool2d
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import initialization as init

import warnings
warnings.filterwarnings("ignore")

class AneurysmModel(nn.Module):
    def __init__(self, model_name, pretrained, image_size, num_classes):
        super(AneurysmModel, self).__init__()
        if 'convnext' in model_name or 'efficientnet' in model_name or str(image_size) in model_name:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=3, num_classes=num_classes)
        else:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=3, img_size=image_size, num_classes=num_classes)
    
    @autocast(device_type='cuda')
    def forward(self, x):
        return self.backbone(x)

class AneurysmAuxModel(nn.Module):
    def __init__(self, encoder_name=None, decoder_name=None, encoder_weights=None, encoder_feat_dims=None, num_classes=14, test_mode=False):
        super(AneurysmAuxModel, self).__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )
        self.test_mode = test_mode
        self.encoder_feat_dims = encoder_feat_dims

        if 'mit_b' in encoder_name:
            if encoder_name == "mit_b0":
                self.conv_head = nn.Conv2d(256, self.encoder_feat_dims, 1, 1, bias=False)
            elif encoder_name in ["mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5"]: 
                self.conv_head = nn.Conv2d(512, self.encoder_feat_dims, 1, 1, bias=False)
            else:
                raise ValueError()
            self.bn2 = BatchNormAct2d(num_features=self.encoder_feat_dims)
            self.global_pool = SelectAdaptivePool2d()
            if decoder_name == 'fpn':
                self.decoder = FPNDecoder(
                    encoder_channels=self.encoder.out_channels,
                    encoder_depth=5,
                    pyramid_channels=256,
                    segmentation_channels=128,
                    dropout=0.2,
                    merge_policy="add",
                )
                self.segmentation_head = SegmentationHead(
                    in_channels=self.decoder.out_channels,
                    out_channels=2,
                    kernel_size=1,
                    upsampling=4,
                    activation=None,
                )
            elif decoder_name == 'unet':
                decoder_channels = (256, 128, 64, 32, 16)
                self.decoder = UnetDecoder(
                    encoder_channels=self.encoder.out_channels,
                    decoder_channels=decoder_channels,
                    n_blocks=5,
                    use_norm="batchnorm",
                    add_center_block=False,
                    attention_type=None,
                )
                self.segmentation_head = SegmentationHead(
                    in_channels=decoder_channels[-1],
                    out_channels=2,
                    kernel_size=3,
                    activation=None,
                )
            else:
                raise ValueError()

        elif "timm-efficientnet" in encoder_name:
            self.conv_head = self.encoder.conv_head
            self.bn2 = self.encoder.bn2
            self.global_pool = self.encoder.global_pool
            if decoder_name == 'unet':
                decoder_channels = (256, 128, 64, 32, 16)
                self.decoder = UnetDecoder(
                    encoder_channels=self.encoder.out_channels,
                    decoder_channels=decoder_channels,
                    n_blocks=5,
                    use_norm="batchnorm",
                    add_center_block=False,
                    attention_type=None,
                )
                self.segmentation_head = SegmentationHead(
                    in_channels=decoder_channels[-1],
                    out_channels=2,
                    kernel_size=3,
                    activation=None,
                )
            else:
                raise ValueError()

        
        self.fc = nn.Linear(self.encoder_feat_dims, 1024, bias=True)
        self.cls_head = nn.Linear(1024, num_classes, bias=True)

        init.initialize_head(self.fc)
        init.initialize_head(self.cls_head)
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    @autocast(device_type='cuda')
    def forward(self, x):
        x = self.encoder(x)
        y_cls = self.conv_head(x[-1])
        y_cls = self.bn2(y_cls)
        y_cls = self.global_pool(y_cls)
        y_cls = y_cls.view(-1, self.encoder_feat_dims)
        y_cls = self.fc(y_cls)
        y_cls = F.relu(y_cls)
        y_cls = F.dropout(y_cls, p=0.5, training=self.training)
        y_cls = self.cls_head(y_cls)

        if self.test_mode:
            return y_cls
        else:
            y_seg = self.decoder(x)
            y_seg = self.segmentation_head(y_seg)
            return y_cls, y_seg

def dicom2image_process(dcm_path):
    dcm_file = pydicom.dcmread(dcm_path, force=True)
    ImagePositionPatient = dcm_file.get('ImagePositionPatient', [0,0,0])
    z = ImagePositionPatient[2]
    Modality = dcm_file.get('Modality', 'MR')
    SOPInstanceUID = dcm_file.get('SOPInstanceUID', None)
    image = dcm_file.pixel_array.astype(np.float32)
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

    return image, z, Modality, SOPInstanceUID

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

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


class RSNA_IAD:
    def __init__(self,
                 exp0_aneurysm_det_ckpt=None, 
                 exp1_brain_det_ckpt=None,
                 exp2_vit_large_ckpt=None,
                 exp2_eva_large_ckpt=None, 
                 exp3_mit_b4_ckpt=None,
                 exp4_vit_large_ckpt=None,
                 exp4_eva_large_ckpt=None,
                 exp5_mit_b4_ckpt=None,
                 device=None):
        self.classes = [
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

        self.device = device
        print('Loading models...')
        self.exp0_aneurysm_det_model = YOLO(exp0_aneurysm_det_ckpt)
        
        self.exp1_brain_det_model = DetectMultiBackend(exp1_brain_det_ckpt, device=select_device(device), dnn=False, fp16=True)
        self.exp1_brain_det_model.eval()
        self.exp1_brain_det_model.warmup(imgsz=(1, 3, 640, 640))

        self.exp3_mit_b4_model = AneurysmAuxModel(
            encoder_name='mit_b4',
            decoder_name='fpn',
            encoder_weights=None, 
            encoder_feat_dims=1280, 
            num_classes=14,
            test_mode=True
        )
        self.exp3_mit_b4_model.load_state_dict(torch.load(exp3_mit_b4_ckpt, weights_only=True))
        self.exp3_mit_b4_model.to(self.device)
        self.exp3_mit_b4_model.eval()
        
        self.exp5_mit_b4_model = AneurysmAuxModel(
            encoder_name='mit_b4',
            decoder_name='fpn',
            encoder_weights=None, 
            encoder_feat_dims=1280, 
            num_classes=14,
            test_mode=True
        )
        self.exp5_mit_b4_model.load_state_dict(torch.load(exp5_mit_b4_ckpt, weights_only=True))
        self.exp5_mit_b4_model.to(self.device)
        self.exp5_mit_b4_model.eval()

        self.exp2_vit_large_model = AneurysmModel(
            model_name='vit_large_patch14_clip_336.openai_ft_in12k_in1k',
            pretrained=False, 
            image_size=384,
            num_classes=14,
        )
        self.exp2_vit_large_model.load_state_dict(torch.load(exp2_vit_large_ckpt, weights_only=True))
        self.exp2_vit_large_model.to(self.device)
        self.exp2_vit_large_model.eval()

        self.exp4_vit_large_model = AneurysmModel(
            model_name='vit_large_patch14_clip_336.openai_ft_in12k_in1k',
            pretrained=False, 
            image_size=384,
            num_classes=14,
        )
        self.exp4_vit_large_model.load_state_dict(torch.load(exp4_vit_large_ckpt, weights_only=True))
        self.exp4_vit_large_model.to(self.device)
        self.exp4_vit_large_model.eval()

        self.exp2_eva_large_model = AneurysmModel(
            model_name='eva_large_patch14_336.in22k_ft_in22k_in1k',
            pretrained=False, 
            image_size=384,
            num_classes=14,
        )
        self.exp2_eva_large_model.load_state_dict(torch.load(exp2_eva_large_ckpt, weights_only=True))
        self.exp2_eva_large_model.to(self.device)
        self.exp2_eva_large_model.eval()

        self.exp4_eva_large_model = AneurysmModel(
            model_name='eva_large_patch14_336.in22k_ft_in22k_in1k',
            pretrained=False, 
            image_size=384,
            num_classes=14,
        )
        self.exp4_eva_large_model.load_state_dict(torch.load(exp4_eva_large_ckpt, weights_only=True))
        self.exp4_eva_large_model.to(self.device)
        self.exp4_eva_large_model.eval()

    def predict(self, series_path=None, batch_size=8, crop_rat=0.75, aneurysm_thres=0.5, aneurysm_det_thres=0.2):
        dcm_filepaths = []
        for root, _, files in os.walk(series_path):
            for file in files:
                if file.endswith('.dcm'):
                    dcm_filepaths.append(os.path.join(root, file))

        images = []
        z_pos = []
        modalities = []
        SOPInstanceUIDs = []
        for dcm_path in dcm_filepaths:
            image, z, Modality, SOPInstanceUID = dicom2image_process(dcm_path)
            if image.ndim == 2:
                images.append(image)
                z_pos.append(z)
                modalities.append(Modality)
            elif image.ndim == 3:
                for i in range(image.shape[0]):
                    image_i = image[i,:,:]
                    images.append(image_i)
                    z_pos.append(i)
                    modalities.append(Modality)
            del image
            SOPInstanceUIDs.append(SOPInstanceUID)
        images = np.array(images)
        z_pos = np.array(z_pos)
        modalities = np.array(modalities)
        SOPInstanceUIDs = np.array(SOPInstanceUIDs)
    
        brain_image = np.mean(images, 0)
        brain_image_min = np.min(brain_image)
        brain_image_max = np.max(brain_image)
        brain_image = (brain_image - brain_image_min) / (brain_image_max - brain_image_min + 1e-7)
        brain_image = (brain_image * 255).astype(np.uint8)

        brain_height, brain_width = brain_image.shape[0:2]
        brain_image = np.stack([brain_image, brain_image, brain_image], -1)

        brain_image = img2tensor_yolov5(brain_image, imgsz=640, stride=32, device=select_device(self.device))
        boxes = []
        labels = []
        scores = []
        with autocast("cuda"), torch.no_grad():
            brain_dets = self.exp1_brain_det_model(brain_image, augment=False, visualize=False)
            brain_dets = non_max_suppression(brain_dets, conf_thres=0.1, iou_thres=0.4, classes=None, agnostic=True, max_det=100)
    
            for det in brain_dets:
                if len(det):
                    det[:, :4] = scale_boxes(brain_image.shape[2:], det[:, :4], (brain_height, brain_width)).round()
                    det = det.data.cpu().numpy()
                    for d in det:
                        x1, y1, x2, y2, score, label = int(d[0]), int(d[1]), int(d[2]), int(d[3]), float(d[4]), int(d[5])
                        x1 = min(max(0, x1), brain_width)
                        x2 = min(max(0, x2), brain_width)
                        y1 = min(max(0, y1), brain_height)
                        y2 = min(max(0, y2), brain_height)
                        if x1 >= x2 or y1 >= y2:
                            continue
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
            brain_x1, brain_y1, brain_x2, brain_y2, brain_class_name = 0, 0, brain_width, brain_height, 'brain'
        
        if crop_rat != 1:
            brain_xc = 0.5*(brain_x1+brain_x2)
            brain_yc = 0.5*(brain_y1+brain_y2)
            brain_width = crop_rat*(brain_x2-brain_x1)
            brain_height = crop_rat*(brain_y2-brain_y1)
            crop_x1 = int(brain_xc - 0.5*brain_width)
            crop_x2 = int(brain_xc + 0.5*brain_width)
            crop_y1 = int(brain_yc - 0.5*brain_height)
            crop_y2 = int(brain_yc + 0.5*brain_height)
        else:
            crop_x1, crop_y1, crop_x2, crop_y2 = brain_x1, brain_y1, brain_x2, brain_y2

        idxs = np.argsort(z_pos)
        z_pos = z_pos[idxs]
        images = images[idxs,:,:]
        modalities = modalities[idxs]
        SOPInstanceUIDs = SOPInstanceUIDs[idxs]
        
        images_tensor = images[:,crop_y1:crop_y2,crop_x1:crop_x2]
        images_tensor = torch.from_numpy(images_tensor).unsqueeze(1).type(torch.float16).to(self.device)
        images_tensor = F.interpolate(images_tensor, size=(384, 384), mode='bilinear', align_corners=True).squeeze().type(torch.float16).cpu()
        images_tensor /= 255.0

        ser_preds = []
        for start in range(0, images_tensor.shape[0], batch_size):
            end = min(start+batch_size, images_tensor.shape[0])
            batch_image = []
            for i in range(start, end, 1):
                if i == 0:
                    pre_i = i
                else:
                    pre_i = i-1
                if i == images_tensor.size(0) - 1:
                    next_i = i
                else:
                    next_i = i+1
                image = images_tensor[[pre_i, i, next_i],:,:]
                batch_image.append(image)
            batch_image = torch.stack(batch_image, 0).to(self.device)

            with autocast("cuda"), torch.no_grad():
                exp3_mit_b4_batch_pred = torch.sigmoid(self.exp3_mit_b4_model(batch_image))
                exp3_mit_b4_batch_pred = exp3_mit_b4_batch_pred.data.cpu().numpy()

                exp5_mit_b4_batch_pred = torch.sigmoid(self.exp5_mit_b4_model(batch_image))
                exp5_mit_b4_batch_pred = exp5_mit_b4_batch_pred.data.cpu().numpy()

                exp2_vit_large_batch_pred = torch.sigmoid(self.exp2_vit_large_model(batch_image))
                exp2_vit_large_batch_pred = exp2_vit_large_batch_pred.data.cpu().numpy()

                exp4_vit_large_batch_pred = torch.sigmoid(self.exp4_vit_large_model(batch_image))
                exp4_vit_large_batch_pred = exp4_vit_large_batch_pred.data.cpu().numpy()

                exp2_eva_large_batch_pred = torch.sigmoid(self.exp2_eva_large_model(batch_image))
                exp2_eva_large_batch_pred = exp2_eva_large_batch_pred.data.cpu().numpy()

                exp4_eva_large_batch_pred = torch.sigmoid(self.exp4_eva_large_model(batch_image))
                exp4_eva_large_batch_pred = exp4_eva_large_batch_pred.data.cpu().numpy()

                batch_pred = 0.25*exp3_mit_b4_batch_pred + \
                             0.25*exp5_mit_b4_batch_pred + \
                             0.125*exp2_vit_large_batch_pred + \
                             0.125*exp2_eva_large_batch_pred + \
                             0.125*exp4_vit_large_batch_pred + \
                             0.125*exp4_eva_large_batch_pred

                ser_preds.append(batch_pred)

        ser_preds = np.concatenate(ser_preds, 0)
    
        ### show slice and bounding box with highest aneurysm_score if aneurysm_score > thres else return None
        if np.max(ser_preds[:,-1]) > aneurysm_thres:
            aneurysm_centroid_slice_index = np.argmax(ser_preds[:,-1])
            aneurysm_centroid_slice_image = images[aneurysm_centroid_slice_index,:,:]
            aneurysm_centroid_SOPInstanceUID = SOPInstanceUIDs[aneurysm_centroid_slice_index]

            if aneurysm_centroid_slice_index == 0:
                pre_i = aneurysm_centroid_slice_index
            else:
                pre_i = aneurysm_centroid_slice_index-1
            if aneurysm_centroid_slice_index == images.shape[0] - 1:
                next_i = aneurysm_centroid_slice_index
            else:
                next_i = aneurysm_centroid_slice_index+1
            exp0_aneurysm_det_image = images[[pre_i, aneurysm_centroid_slice_index, next_i],:,:]
            exp0_aneurysm_det_image = np.transpose(exp0_aneurysm_det_image, (1,2,0))
                
            det = self.exp0_aneurysm_det_model(exp0_aneurysm_det_image, imgsz=1280, conf=aneurysm_det_thres, iou=0.4, agnostic_nms=True, augment=False, device=self.device, verbose=False)[0]
            boxes = det.boxes.xyxy.data.cpu().numpy().astype(int)

            aneurysm_centroid_slice_image = np.stack([aneurysm_centroid_slice_image, aneurysm_centroid_slice_image, aneurysm_centroid_slice_image], -1)
            for box in boxes:
                exp0_aneurysm_x1, exp0_aneurysm_y1, exp0_aneurysm_x2, exp0_aneurysm_y2 = box 
                cv2.rectangle(aneurysm_centroid_slice_image, (exp0_aneurysm_x1, exp0_aneurysm_y1), (exp0_aneurysm_x2, exp0_aneurysm_y2), (255,0,0), 1)
            
            localizer = {
                'image': aneurysm_centroid_slice_image,
                'SOPInstanceUID': aneurysm_centroid_SOPInstanceUID
            }
        else:
            localizer = None

        ser_preds = np.max(ser_preds, 0)
        ser_preds = dict(zip(self.classes, ser_preds))
        return ser_preds, localizer