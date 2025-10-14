import argparse
import yaml
from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default='config/yolo11x_1280.yaml', type=str)
parser.add_argument("--fold", default=0, type=int)
args = parser.parse_args()
print(args)

if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)

    model = YOLO('{}.pt'.format(cfg['model_name']))
    results = model.train(
        data='data/data_fold{}.yaml'.format(args.fold), 
        project="checkpoints",
        imgsz=cfg['image_size'], 
        name="{}_{}_fold{}".format(cfg['model_name'], cfg['image_size'], args.fold), 
        epochs=cfg['epochs'], 
        batch=cfg['batch_size'], 
        workers=cfg['workers'],
        flipud=0.5, 
        fliplr=0.5,
        mixup=0,
        mosaic=0,
    )