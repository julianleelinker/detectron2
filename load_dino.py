import torch
from detectron2.utils import custom

backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()

output_list = custom.register_top_outputs(backbone)

x0 = torch.rand((1, 3, 224, 224)).cuda()
x1 = backbone(x0)

# for (name, feat) in output_list:
#     print(name, feat.shape)

import ipdb; ipdb.set_trace()