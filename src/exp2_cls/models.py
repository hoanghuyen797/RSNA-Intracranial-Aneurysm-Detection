import torch.nn as nn
import timm
from torch.amp import autocast

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
