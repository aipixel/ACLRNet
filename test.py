import os
import cv2
import argparse
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import transforms
from torchvision.transforms import ToTensor
from model import *
from skimage.metrics import structural_similarity as sk_cpt_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_cpt_psnr


def cal_psnr(img1, img2):
    """
    Calculate psnr of the img1 and the img2.
    :param img1: numpy array
    :param img2: numpy array
    :return: np.float32
    """
    return sk_cpt_psnr(img1, img2)

def cal_ssim(img1, img2):
    """
    Calculate ssim of the img1 and the img2.
    :param img1: numpy array
    :param img2: numpy array
    :return: np.float32
    """
    img1 = img1.transpose((1, 2, 0))
    img2 = img2.transpose((1, 2, 0))
    return sk_cpt_ssim(img1, img2, data_range=1, multichannel=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--testset_dir', type=str, default=r'data/test/KITTI2012',
                        help='KITTI2012 or KITTI2015 or Middlebury or ETH3D or Flickr1024')
    parser.add_argument('--model_name', type=str, default=None, help='model ckpt name')
    parser.add_argument('--task', type=str, default='DN', help='SR or DN or CAR')
    parser.add_argument('--scale_factor', type=int, default=4, help='SR scale factor 2 or 4')
    parser.add_argument('--noise_level', type=int, default=30, help='DN noise level 10 or 30')
    parser.add_argument('--quality_factor', type=int, default=30, help='CAR quality factor 10 or 30')
    parser.add_argument('--save_result', type=bool, default=False)
    return parser.parse_args()

def test(cfg):
    if cfg.task == 'SR':
        net = Net(upscale_factor=cfg.scale_factor, in_nc=3, out_nc=3, ng0=64, ng=24, nbc=4, nb=2).to(cfg.device)
        path_name = "lr_x" + str(cfg.scale_factor)
    elif cfg.task == 'DN':
        net = Net(upscale_factor=1, in_nc=3, out_nc=3, ng0=64, ng=24, nbc=4, nb=2).to(cfg.device)
        path_name = "noise_" + str(cfg.noise_level)
    elif cfg.task == 'CAR':
        net = Net(upscale_factor=1, in_nc=1, out_nc=1, ng0=64, ng=24, nbc=4, nb=2).to(cfg.device)
        path_name = "jpeg_" + str(cfg.quality_factor)
    else:
        raise ValueError('task should be SR or DN or CAR')

    if cfg.model_name:
        model = torch.load("./ckpt/" + cfg.model_name + ".pth.tar", map_location=cfg.device)
        #net.load_state_dict(model['state_dict'])
        new_state_dict = OrderedDict()
        for k, v in model['state_dict'].items():
            name = k[7:]
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    net.eval()

    psnr_list = []
    ssim_list = []
    for idx, img_name in enumerate(os.listdir(cfg.testset_dir + '/GT')[::2]):
        if cfg.task == 'CAR':
            HR_left = cv2.imread(cfg.testset_dir + '/jpeg_100/' + str(idx+1).zfill(4) + '_L.png', cv2.IMREAD_GRAYSCALE)
            HR_right = cv2.imread(cfg.testset_dir + '/jpeg_100/' + str(idx+1).zfill(4) + '_R.png', cv2.IMREAD_GRAYSCALE)
            LR_left = cv2.imread(cfg.testset_dir + '/' + path_name + '/' + str(idx+1).zfill(4) + '_L.png', cv2.IMREAD_GRAYSCALE)
            LR_right = cv2.imread(cfg.testset_dir + '/' + path_name + '/' + str(idx+1).zfill(4) + '_R.png', cv2.IMREAD_GRAYSCALE)
        else:
            HR_left = cv2.imread(cfg.testset_dir + '/GT/' + str(idx+1).zfill(4) + '_L.png')
            HR_right = cv2.imread(cfg.testset_dir + '/GT/' + str(idx+1).zfill(4) + '_R.png')
            LR_left = cv2.imread(cfg.testset_dir + '/' + path_name + '/' + str(idx+1).zfill(4) + '_L.png')
            LR_right = cv2.imread(cfg.testset_dir + '/' + path_name + '/' + str(idx+1).zfill(4) + '_R.png')

        LR_left, LR_right = ToTensor()(LR_left), ToTensor()(LR_right)
        HR_left, HR_right = ToTensor()(HR_left), ToTensor()(HR_right)

        LR_left, LR_right = LR_left.unsqueeze(0), LR_right.unsqueeze(0)
        HR_left, HR_right = HR_left.unsqueeze(0), HR_right.unsqueeze(0)

        HR_left, HR_right, LR_left, LR_right = Variable(HR_left).to(cfg.device), Variable(HR_right).to(cfg.device), \
                                               Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)

        with torch.no_grad():
            SR_left, SR_right = net(LR_left, LR_right)
            SR_left, SR_right = torch.clamp(SR_left, 0, 1), torch.clamp(SR_right, 0, 1)

        #SR
        if SR_left.shape != HR_left.shape:
            SR_left = F.interpolate(SR_left, size=(HR_left.shape[2], HR_left.shape[3]), mode='bicubic')
            SR_right = F.interpolate(SR_right, size=(HR_right.shape[2], HR_right.shape[3]), mode='bicubic')

        psnr_l = cal_psnr(HR_left[:, :, :, :].data.cpu().numpy(), SR_left[:, :, :, :].data.cpu().numpy())
        ssim_l = cal_ssim(HR_left[0, :, :, :].data.cpu().numpy(), SR_left[0, :, :, :].data.cpu().numpy())

        psnr_r = cal_psnr(HR_right[:, :, :, :].data.cpu().numpy(), SR_right[:, :, :, :].data.cpu().numpy())
        ssim_r = cal_ssim(HR_right[0, :, :, :].data.cpu().numpy(), SR_right[0, :, :, :].data.cpu().numpy())

        psnr_list.append(psnr_l)
        ssim_list.append(ssim_l)
        psnr_list.append(psnr_r)
        ssim_list.append(ssim_r)

        print('{}, psnrl: {}, ssiml: {}, psnrr: {}, ssimr: {}'.format(str(idx+1).zfill(4), psnr_l, ssim_l, psnr_r, ssim_r))

        if cfg.save_result:
            save_path = './results/' + cfg.dataset
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
            SR_left_img.save(save_path + '/' + str(idx+1).zfill(4) + '_L.png')
            SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
            SR_right_img.save(save_path + '/' + str(idx+1).zfill(4) + '_R.png')

    print('Avg. PSNR: {:.5f} dB, Avg. SSIM: {:.5f}'.format(np.mean(psnr_list), np.mean(ssim_list)))


if __name__ == '__main__':
    cfg = parse_args()
    cfg.dataset = 'val'
    test(cfg)
    print('Finished!')
