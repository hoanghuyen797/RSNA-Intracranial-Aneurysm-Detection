import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from timm.layers import BatchNormAct2d, SelectAdaptivePool2d
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import initialization as init

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