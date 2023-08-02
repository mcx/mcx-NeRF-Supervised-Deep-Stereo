import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('dataloaders')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import cv2

# From https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# For RAFT-Stereo
sys.path.append(os.path.join(parentdir,'code_snippets/core'))
# print(sys.path)

from models.PSMNet.models import basic, stackhourglass
from raft_stereo import RAFTStereo

def load_pretrained_model(args):
    print('Load pretrained model')
    model = None
    if args.model == 'raft-stereo':
        model = RAFTStereo(args)
    elif args.model == 'psmnet':
        model = stackhourglass(args.maxdisp)
    else:
        print('Invalid model selected.')
        exit()

    if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()

    if args.loadmodel is not None:
        pretrain_dict = torch.load(args.loadmodel, torch.device('cuda:0'))
        model.load_state_dict(pretrain_dict)
    else:
        print('A pretrained model is required!')

    return model

@torch.no_grad()
def run(model, left, right, args):
    model.eval()

    if args.cuda:
        left, right = left.cuda(), right.cuda()

    pad_ht = (((left.shape[-2] // 32) + 1) * 32 - left.shape[-2]) % 32
    pad_wd = (((left.shape[-1] // 32) + 1) * 32 - left.shape[-1]) % 32

    _pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    if args.model == 'raft-stereo':
        left = F.pad(left, _pad, mode='replicate')
        right = F.pad(right, _pad, mode='replicate')
    else:
        left = F.pad(left, _pad)
        right = F.pad(right, _pad)

    pred_disps = model(left, right)

    if args.model == 'psmnet':
        pred_disp = pred_disps[0]
    elif args.model == 'raft-stereo':
        pred_disp = pred_disps[-1].squeeze()

    ht, wd = pred_disp.shape[-2:]
    c = [_pad[2], ht - _pad[3], _pad[0], wd - _pad[1]]
    pred_disp = pred_disp[..., c[0]:c[1], c[2]:c[3]]

    return pred_disp.squeeze()

def main():
    parser = argparse.ArgumentParser(description='Stereo Depth Estimation Demo')
    parser.add_argument('--maxdisp', type=int, default=256, help='maximum disparity')
    parser.add_argument('--model', default='raft-stereo', choices=['psmnet', 'raft-stereo'], help='select model')
    parser.add_argument('--loadmodel', default='./weights/raftstereo-NS.tar', help='load model')
    parser.add_argument('--left', help='left image path')
    parser.add_argument('--right', help='right image path')
    parser.add_argument('--output', help='output path for predicted disparity')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA')

    # RAFT-Stereo
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    model = load_pretrained_model(args)

    left_img = cv2.imread(args.left, 1)
    right_img = cv2.imread(args.right, 1)

    if left_img is None or right_img is None:
        print('Failed to read image(s).')
        exit()

    left_img = np.transpose(left_img, (2, 0, 1))
    right_img = np.transpose(right_img, (2, 0, 1))

    left_img = np.expand_dims(left_img, 0)
    right_img = np.expand_dims(right_img, 0)

    if args.cuda:
        left_img = torch.from_numpy(left_img).cuda().float()
        right_img = torch.from_numpy(right_img).cuda().float()
    else:
        left_img = torch.from_numpy(left_img).float()
        right_img = torch.from_numpy(right_img).float()

    pred_disp = run(model, left_img, right_img, args)

    plt.imsave(args.output, pred_disp.cpu().numpy(), cmap='magma')

    print("Done!")

if __name__ == '__main__':
    main()

