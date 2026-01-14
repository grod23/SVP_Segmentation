from monai.networks.nets import (
    UNet, AttentionUnet, SegResNetDS, DynUNet, BasicUNetPlusPlus, FlexibleUNet,
    UNETR, SwinUNETR
)
import torch.nn as nn
from src import IMAGE_SIZE, IN_CHANNELS, OUT_CHANNELS

class Backbone(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        if self.backbone_name == 'unet':
            self.backbone = UNet(
                spatial_dims=2,
                in_channels=IN_CHANNELS,
                out_channels=OUT_CHANNELS,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2
            )
        elif self.backbone_name == 'attentionunet':
            self.backbone = AttentionUnet(
                spatial_dims=2,
                in_channels=IN_CHANNELS,
                out_channels=OUT_CHANNELS,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2)
            )
        elif self.backbone_name == 'segresnetds':
            self.backbone = SegResNetDS(
                spatial_dims=2,
                in_channels=IN_CHANNELS,
                out_channels=OUT_CHANNELS,
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1]
            )
        elif self.backbone_name == 'dynunet':
            self.backbone = DynUNet(
                spatial_dims=2,
                in_channels=IN_CHANNELS,
                out_channels=OUT_CHANNELS,
                kernel_size=[[3,3]]*4,
                strides=[[2,2]]*4,
                filters=[16, 32, 64, 128, 256]
            )
        elif self.backbone_name == 'basicunetplusplus':
            self.backbone = BasicUNetPlusPlus(
                spatial_dims=2,
                in_channels=IN_CHANNELS,
                out_channels=OUT_CHANNELS,
                features=(16, 32, 64, 128)
            )
        elif self.backbone_name == 'flexibleunet':
            self.backbone = FlexibleUNet(
                spatial_dims=2,
                in_channels=IN_CHANNELS,
                out_channels=OUT_CHANNELS
            )


class Segmentation_Model(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        self.backbone = Backbone(backbone_name)


    def forward(self, X_image):
        y = self.backbone(X_image)
        return y

