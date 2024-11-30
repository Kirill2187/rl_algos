import torch.nn as nn


def build_network(input_size, output_size, architecture_config, activation_fn=None):
    layers = []
    in_features = input_size
    for layer_cfg in architecture_config:
        layer_type = layer_cfg['type']

        if layer_type == 'Linear':
            out_features = layer_cfg['out_features']
            layers.append(nn.Linear(in_features, out_features))
            in_features = out_features
        elif layer_type == 'ReLU':
            layers.append(nn.ReLU())
        elif layer_type == 'Tanh':
            layers.append(nn.Tanh())
        elif layer_type == 'Sigmoid':
            layers.append(nn.Sigmoid())
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    layers.append(nn.Linear(in_features, output_size))

    if activation_fn is not None:
        layers.append(activation_fn)

    return nn.Sequential(*layers)
