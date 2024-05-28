import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone import BasicStem, FPN, ResNet
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import (
    StandardROIHeads,
    FastRCNNOutputLayers,
    FastRCNNConvFCHead,
)

from ..data.constants import constants
from detectron2.modeling import DinoViT


# Small 
embed_dim, depth, num_heads, dp = 384, 12, 6, 0.1

image_size = 14*37

model = L(GeneralizedRCNN)(
    backbone = L(DinoViT)(  # Single-scale ViT backbone
        img_size=image_size,
        patch_size=14,
        #in_chans=3,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        # ffn_bias=True,
        # proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=1.0,  # for layerscale: None or 0 => no layerscale
        #embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        #block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        out_feature="last_feat",
        square_pad=image_size,
    ),
    proposal_generator=L(RPN)(
        # in_features=["p2", "p3", "p4", "p5", "p6"],
        in_features=["last_feat"],
        head=L(StandardRPNHead)(in_channels=embed_dim, num_anchors=3),
        anchor_generator=L(DefaultAnchorGenerator)(
            sizes=[[32], [64], [128], [256], [512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            offset=0.0,
        ),
        anchor_matcher=L(Matcher)(
            thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
        ),
        box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(2000, 1000),
        post_nms_topk=(1000, 1000),
        nms_thresh=0.7,
    ),
    roi_heads=L(StandardROIHeads)(
        num_classes=23,
        batch_size_per_image=512,
        positive_fraction=0.25,
        proposal_matcher=L(Matcher)(
            thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
        ),
        # box_in_features=["p2", "p3", "p4", "p5"],
        box_in_features=["last_feat"],
        box_pooler=L(ROIPooler)(
            output_size=7,
            # scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            scales=(1.0 / 16,),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        box_head=L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=embed_dim, height=7, width=7),
            conv_dims=[],
            fc_dims=[1024, 1024],
        ),
        box_predictor=L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
            num_classes="${..num_classes}",
        ),
        # mask_in_features=["p2", "p3", "p4", "p5"],
        # mask_pooler=L(ROIPooler)(
        #     output_size=14,
        #     scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
        #     sampling_ratio=0,
        #     pooler_type="ROIAlignV2",
        # ),
        # mask_head=L(MaskRCNNConvUpsampleHead)(
        #     input_shape=ShapeSpec(channels=256, width=14, height=14),
        #     num_classes="${..num_classes}",
        #     conv_dims=[256, 256, 256, 256, 256],
        # ),
    ),
    pixel_mean=constants.imagenet_bgr256_mean,
    pixel_std=constants.imagenet_bgr256_std,
    input_format="BGR",
)

model.roi_heads.box_head.conv_norm = "LN"

# 2conv in RPN:
model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]