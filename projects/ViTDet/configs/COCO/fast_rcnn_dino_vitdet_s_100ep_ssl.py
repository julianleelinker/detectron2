from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from ..common.coco_loader_ssl import dataloader

dataloader.train.total_batch_size = 4

model = model_zoo.get_config("common/models/fast_rcnn_dino_vitdet_b.py").model
model.backbone.net.embed_dim = 384
model.backbone.net.depth = 12
model.backbone.net.num_heads = 6
model.backbone.net.patch_size = 16

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = (
    "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
    # "/home/appuser/detectron2_repo/datasets/coco/dinov2_vitb14_reg4_pretrain.pth"
    # "/home/appuser/detectron2/model-weights/dinov2_vits14_pretrain.pth"
)
train.output_dir = '/home/appuser/datasets/tiip/train-output/test'
train.ddp.find_unused_parameters = True

# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
num_epoch = 400
num_train_images = 10209
train.max_iter = num_epoch*num_train_images // dataloader.train.total_batch_size
train.eval_period = num_train_images // dataloader.train.total_batch_size

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[163889, 177546],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}