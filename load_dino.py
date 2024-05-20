import torch

# DINOv2
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()

features = []
def named_hook(name):
    def hook(module, input, output): 
        features.append((name, output.clone().detach()))
    return hook

outer_layers = []
for name, module in backbone.named_modules():
    if len(name.split('.'))!=1 or name=='':
        continue
    if isinstance(module, torch.nn.ModuleList):
        outer_layers.append((f'{name}.{len(module)}', module[-1]))
    outer_layers.append((name, module))

for name, module in outer_layers:
    handle = module.register_forward_hook(named_hook(name))

x0 = torch.rand((1, 3, 224, 224)).cuda()
x1 = backbone(x0)

for (name, feat) in features:
    print(name, feat.shape)

import ipdb; ipdb.set_trace()