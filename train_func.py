import torch
import deepnovo_config
from model import DeepNovoModel, device, InitNet
from deepnovo_config import args

def build_model(training=True):
    forward_deepnovo = DeepNovoModel()
    backward_deepnovo = DeepNovoModel()
    if deepnovo_config.use_lstm:
        init_net = InitNet()
    else:
        init_net = None
    forward_deepnovo.load_state_dict(torch.load(args.forward_model,map_location=device))
    backward_deepnovo.load_state_dict(torch.load(args.backward_model,map_location=device))
    if deepnovo_config.use_lstm:
        init_net.load_state_dict(torch.load(args.init_model,map_location=device))
    if deepnovo_config.use_lstm:
        backward_deepnovo.embedding.weight = forward_deepnovo.embedding.weight
    backward_deepnovo = backward_deepnovo.to(device)
    forward_deepnovo = forward_deepnovo.to(device)
    if deepnovo_config.use_lstm:
        init_net = init_net.to(device)
    return forward_deepnovo, backward_deepnovo, init_net