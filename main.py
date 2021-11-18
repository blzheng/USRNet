import os.path
import cv2
import logging
import argparse
import numpy as np
from datetime import datetime
from collections import OrderedDict
from scipy.io import loadmat
from scipy import ndimage
import scipy.io as scio

import torch
import time

from utils import utils_deblur
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util

#from models.network_usrnet import USRNet as net   # for pytorch version <= 1.7.1
# from models.network_usrnet_v1 import USRNet as net  # for pytorch version >= 1.8.1
from models.network_usrnet_v2 import USRNet as net  # for pytorch version >= 1.8.1
from torch.profiler import profile, record_function, ProfilerActivity

'''
Spyder (Python 3.7)
PyTorch 1.4.0
Windows 10 or Linux

Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/USRNet
        https://github.com/cszn/KAIR

If you have any question, please feel free to contact with me.
Kai Zhang (e-mail: cskaizhang@gmail.com)

by Kai Zhang (12/March/2020)
'''

"""
# --------------------------------------------
testing code of USRNet for the Table 1 in the paper
@inproceedings{zhang2020deep,
  title={Deep unfolding network for image super-resolution},
  author={Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3217--3226},
  year={2020}
}
# --------------------------------------------
|--model_zoo                # model_zoo
   |--usrgan                # model_name, optimized for perceptual quality
   |--usrnet                # model_name, optimized for PSNR
   |--usrgan_tiny           # model_name, tiny model optimized for perceptual quality
   |--usrnet_tiny           # model_name, tiny model optimized for PSNR
|--testsets                 # testsets
   |--real_set              # testset_name, contain 3 real images
|--results                  # results
   |--real_set_usrnet       # result_name = testset_name + '_' + model_name
   |--real_set_usrnet_tiny
# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    model_name = 'usrnet'      # 'usrgan' | 'usrnet' | 'usrgan_tiny' | 'usrnet_tiny'
    testset_name = 'set_real'  # test set,  'set_real'
    test_image = 'chip.png'    # 'chip.png', 'comic.png'
    #test_image = 'comic.png'

    sf = 2                     # scale factor, only from {1, 2, 3, 4}
    show_img = False           # default: False
    save_E = True              # save estimated image
    save_LE = True             # save zoomed LR, Estimated images

    # ----------------------------------------
    # set noise level and kernel
    # ----------------------------------------
    if 'chip' in test_image:
        noise_level_img = 15       # noise level for LR image, 15 for chip
        kernel_width_default_x1234 = [0.6, 0.9, 1.7, 2.2] # Gaussian kernel widths for x1, x2, x3, x4
    else:
        noise_level_img = 2       # noise level for LR image, 0.5~3 for clean images
        kernel_width_default_x1234 = [0.4, 0.7, 1.5, 2.0] # default Gaussian kernel widths of clean/sharp images for x1, x2, x3, x4

    noise_level_model = noise_level_img/255.  # noise level of model
    kernel_width = kernel_width_default_x1234[sf-1]

    # set your own kernel width
    # kernel_width = 2.2

    k = utils_deblur.fspecial('gaussian', 25, kernel_width)
    k = sr.shift_pixel(k, sf)  # shift the kernel
    k /= np.sum(k)
    util.surf(k) if show_img else None
    # scio.savemat('kernel_realapplication.mat', {'kernel':k})

    # load approximated bicubic kernels
    #kernels = hdf5storage.loadmat(os.path.join('kernels', 'kernel_bicubicx234.mat'))['kernels']
#    kernels = loadmat(os.path.join('kernels', 'kernel_bicubicx234.mat'))['kernels']
#    kernel = kernels[0, sf-2].astype(np.float64)

    kernel = util.single2tensor4(k[..., np.newaxis])


    n_channels = 1 if 'gray' in  model_name else 3  # 3 for color image, 1 for grayscale image
    model_pool = 'model_zoo'  # fixed
    testsets = 'testsets'     # fixed
    results = 'results'       # fixed
    result_name = testset_name + '_' + model_name
    model_path = os.path.join(model_pool, model_name+'.pth')

    # ----------------------------------------
    # L_path, E_path
    # ----------------------------------------
    L_path = os.path.join(testsets, testset_name) # L_path, fixed, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, fixed, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------
    if 'tiny' in model_name:
        model = net(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
                    nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
    else:
        model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
                    nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for key, v in model.named_parameters():
        v.requires_grad = False

    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))

    logger.info('model_name:{}, image sigma:{}'.format(model_name, noise_level_img))
    logger.info(L_path)

    img = os.path.join(L_path, test_image)
    # ------------------------------------
    # (1) img_L
    # ------------------------------------
    img_name, ext = os.path.splitext(os.path.basename(img))
    img_L = util.imread_uint(img, n_channels=n_channels)
    img_L = util.uint2single(img_L)

    util.imshow(img_L) if show_img else None
    w, h = img_L.shape[:2]
    logger.info('{:>10s}--> ({:>4d}x{:<4d})'.format(img_name+ext, w, h))

    # boundary handling
    boarder = 8     # default setting for kernel size 25x25
    img = cv2.resize(img_L, (sf*h, sf*w), interpolation=cv2.INTER_NEAREST)
    img = utils_deblur.wrap_boundary_liu(img, [int(np.ceil(sf*w/boarder+2)*boarder), int(np.ceil(sf*h/boarder+2)*boarder)])
    img_wrap = sr.downsample_np(img, sf, center=False)
    img_wrap[:w, :h, :] = img_L
    img_L = img_wrap

    util.imshow(util.single2uint(img_L), title='LR image with noise level {}'.format(noise_level_img)) if show_img else None

    img_L = util.single2tensor4(img_L)
    img_L = img_L.to(device)

    # ------------------------------------
    # (2) img_E
    # ------------------------------------
    sigma = torch.tensor(noise_level_model).float().view([1, 1, 1, 1])
    [img_L, kernel, sigma] = [el.to(device) for el in [img_L, kernel, sigma]]

    img_E = model(img_L, kernel, sf, sigma)

    img_E = util.tensor2uint(img_E)[:sf*w, :sf*h, ...]

    if save_E:
        util.imsave(img_E, os.path.join(E_path, img_name+'_x'+str(sf)+'_'+model_name+'.png'))

    # --------------------------------
    # (3) save img_LE
    # --------------------------------
    if save_LE:
        k_v = k/np.max(k)*1.2
        k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
        k_factor = 3
        k_v = cv2.resize(k_v, (k_factor*k_v.shape[1], k_factor*k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
        img_L = util.tensor2uint(img_L)[:w, :h, ...]
        img_I = cv2.resize(img_L, (sf*img_L.shape[1], sf*img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
        img_I[:k_v.shape[0], :k_v.shape[1], :] = k_v
        util.imshow(np.concatenate([img_I, img_E], axis=1), title='LR / Recovered') if show_img else None
        util.imsave(np.concatenate([img_I, img_E], axis=1), os.path.join(E_path, img_name+'_x'+str(sf)+'_'+model_name+'_LE.png'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def inference(args):
    batch_time = AverageMeter('Time', ':6.3f')
    model_pool = 'model_zoo'  # fixed
    model_path = os.path.join(model_pool, args.model+'.pth')
    datatype = torch.float32
    if args.bf16:
        datatype = torch.bfloat16
    batch_size = args.batch_size

    sf = args.scale_factor                     # scale factor, only from {1, 2, 3, 4}
    # ----------------------------------------
    # set noise level and kernel
    # ----------------------------------------
    noise_level_img = 2       # noise level for LR image, 0.5~3 for clean images
    kernel_width_default_x1234 = [0.4, 0.7, 1.5, 2.0] # default Gaussian kernel widths of clean/sharp images for x1, x2, x3, x4
    noise_level_model = noise_level_img/255.  # noise level of model
    kernel_width = kernel_width_default_x1234[sf-1]
    k = utils_deblur.fspecial('gaussian', 25, kernel_width)
    k = sr.shift_pixel(k, sf)  # shift the kernel
    k /= np.sum(k)
    kernel = util.single2tensor4(k[..., np.newaxis])

    # ----------------------------------------
    # load model
    # ----------------------------------------
    if 'tiny' in args.model:
        model = net(ssf=sf, n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
                    nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
    else:
        model = net(ssf=sf, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
                    nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    if args.ipex:
        import intel_extension_for_pytorch as ipex
        model = model.to(memory_format=torch.channels_last)
        model = ipex.optimize(model, dtype=datatype, level="O1")
    else:
        if args.jit:
            model = model.to(memory_format=torch.channels_last)
        else:
            from torch.utils import mkldnn as mkldnn_utils  
            model = mkldnn_utils.to_mkldnn(model, dtype=datatype)
    if args.jit:
        x = torch.randn(batch_size, 3, args.height, args.width).to(memory_format=torch.channels_last)
        sigma = torch.full((batch_size, 1, 1, 1), noise_level_model).to(memory_format=torch.channels_last)
        kernel = torch.randn(kernel.shape).to(memory_format=torch.channels_last)
        inputs = (x, kernel, sigma)
        if args.bf16:
            with torch.cpu.amp.autocast(), torch.no_grad():
                model = torch.jit.trace(model, inputs)
            model = torch.jit.freeze(model)
        else:
            with torch.no_grad():
                model = torch.jit.trace(model, inputs)
            model = torch.jit.freeze(model)
    with torch.no_grad():
        for i in range(args.max_iters):
            img_L = torch.randn(batch_size, 3, args.height, args.width)
            if i > args.warmup_iters:
                end = time.time()
            if not args.ipex and not args.jit:
                img_L = img_L.to(datatype)
                sigma = sigma.to(datatype)
                kernel = kernel.to(datatype)
            else:
                img_L = img_L.to(memory_format=torch.channels_last)
            if args.ipex and args.bf16 and not args.jit:
                with torch.cpu.amp.autocast():
                    if i == args.warmup_iters:
                        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof, record_function("model_inference"):
                            img_E = model(img_L, kernel, sigma)
                    else:
                        img_E = model(img_L, kernel, sigma)
            else:
                if i == args.warmup_iters:
                    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof, record_function("model_inference"):
                        img_E = model(img_L, kernel, sigma)
                else:
                    img_E = model(img_L, kernel, sigma)
            if i > args.warmup_iters:
                batch_time.update(time.time() - end)
            if args.max_iters != -1 and i >= args.max_iters:
                break
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=-1))
    latency = batch_time.avg / batch_size * 1000
    perf = batch_size / batch_time.avg
    print('Latency: %.3f ms'%latency)
    print("Throughput: {:.3f} fps".format(perf))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='USRNet inference')
    parser.add_argument('--model', type=str, default="usrnet", help='model name: usrgan | usrnet | usrgan_tiny | usrnet_tiny')
    parser.add_argument('--dataset', type=str, default="dummy", help='dataset name')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--ipex', action='store_true', default=False,
                    help='use intel pytorch extension')
    parser.add_argument('--bf16', action='store_true', default=False,
                    help='enable ipex bf16 path')
    parser.add_argument('--jit', action='store_true', default=False,
                    help='enable ipex jit fusionpath')
    parser.add_argument('-w', '--warmup_iters', default=0, type=int, 
                    help='number of warmup iterations to run')
    parser.add_argument('-m', '--max_iters', default=3, type=int, 
                    help='number of max iterations to run')
    parser.add_argument('--height', default=1080, type=int, 
                    help='height of image size')
    parser.add_argument('--width', default=1920, type=int, 
                    help='width of image size')
    parser.add_argument('--scale_factor', default=2, type=int, 
                    help='scale factor, only from {1, 2, 3, 4}')
    parser.add_argument('--instance_id', default=0, type=int, 
                    help='the id of instance')
    parser.add_argument('--num_instances', default=1, type=int, 
                    help='the number of instances')
    parser.add_argument('--num_classes', default=21, type=int,
                    help='the number of classes')
    parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()
    # main()
    inference(args)
