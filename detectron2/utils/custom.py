import torch


def get_top_named_modules(model):
    outer_layers = {}
    for name, module in model.named_modules():
        if len(name.split('.'))!=1 or name=='':
            continue
        if isinstance(module, torch.nn.ModuleList):
            for i in range(len(module)):
                outer_layers[f'{name}.{i}'] = module[i]
        outer_layers[name] = module
    return outer_layers


def named_hook(name, outputs):
    def hook(module, input, output): 
        print(name)
        print('*'*50)
        if isinstance(output, torch.Tensor):
            print(output.shape)
        if isinstance(output, dict):
            for key, value in output.items():
                print(key, value.shape)
        # outputs.append((name, output.clone().detach()))
    return hook


def register_top_outputs(model):
    outer_layers = get_top_named_modules(model)
    output_list = []
    for name, module in outer_layers.items():
        _ = module.register_forward_hook(named_hook(name, output_list))
    return output_list