import argparse
import yaml
from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default='config/yolo11x_1280.yaml', type=str)
args = parser.parse_args()

if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    for fold in range(5):
        model = YOLO('checkpoints/{}_{}_fold{}/weights/best.pt'.format(cfg['model_name'], cfg['image_size'], fold))
        model.val(
            data='data/data_fold{}.yaml'.format(fold), 
            imgsz=cfg['image_size'],
            batch=cfg['batch_size'], 
            workers=cfg['workers']
        )