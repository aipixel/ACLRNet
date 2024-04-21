import torch
import argparse
from utils import *
from model import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.transforms import ToTensor
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
    return sk_cpt_ssim(img1, img2, multichannel=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='optimizer')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--n_epochs', type=int, default=80, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='data/train')
    parser.add_argument('--testset_dir', type=str, default='data/test/KITTI2012/')
    parser.add_argument('--model_name', type=str, default='ACLRNet')
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='log/ACLRNet.pth.tar', help='path to the pretrain model')
    return parser.parse_args()


def train(train_loader, cfg):
    net = Net(upscale_factor=cfg.scale_factor, in_nc=3, out_nc=3, ng0=64, ng=24, nbc=4, nb=2).to(cfg.device)
    net.train()
    cudnn.benchmark = True

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location=cfg.device)
            net.load_state_dict(model['state_dict'])
            cfg.start_epoch = model["epoch"]
        else:
            print("=> no model found at '{}'".format(cfg.load_model))

    criterion_L1 = torch.nn.L1Loss().to(cfg.device)
    criterion_L2 = torch.nn.MSELoss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    loss_epoch = []
    loss_list = []

    for idx_epoch in range(cfg.start_epoch, cfg.n_epochs):
        for idx_iter, (HR_left, HR_right, LR_left, LR_right) in enumerate(train_loader):
            HR_left, HR_right, LR_left, LR_right  = Variable(HR_left).to(cfg.device), Variable(HR_right).to(cfg.device),\
                                                    Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)
            SR_left, SR_right = net(LR_left, LR_right, is_training=1)
            loss = criterion_L2(SR_left, HR_left) + criterion_L2(SR_right, HR_right)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())
        scheduler.step()

        if idx_epoch % 1000 == 0:
            psnr_list = []
            ssim_list = []
            psnr_list1 = []
            ssim_list1 = []
            for idx in range(20):
                LR_left = Image.open(cfg.testset_dir + '/lr_x' + str(cfg.scale_factor) + '/' + str(idx+1).zfill(4) + '_L.png')
                LR_right = Image.open(cfg.testset_dir + '/lr_x' + str(cfg.scale_factor) + '/' + str(idx+1).zfill(4) + '_R.png')
                HR_left = Image.open(cfg.testset_dir + '/GT/' + str(idx + 1).zfill(4) + '_L.png')
                HR_right = Image.open(cfg.testset_dir + '/GT/' + str(idx + 1).zfill(4) + '_R.png')
                LR_left, LR_right = ToTensor()(LR_left), ToTensor()(LR_right)
                HR_left, HR_right = ToTensor()(HR_left), ToTensor()(HR_right)
                LR_left, LR_right = LR_left.unsqueeze(0), LR_right.unsqueeze(0)
                HR_left, HR_right = HR_left.unsqueeze(0), HR_right.unsqueeze(0)
                HR_left, HR_right, LR_left, LR_right = Variable(HR_left).to(cfg.device), Variable(HR_right).to(
                    cfg.device), Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)
                net.eval()
                with torch.no_grad():
                    SR_left, SR_right = net(LR_left, LR_right)
                    SR_left, SR_right = torch.clamp(SR_left, 0, 1), torch.clamp(SR_right, 0, 1)
                if SR_left.shape != HR_left.shape:
                    SR_left = F.interpolate(SR_left, size=(HR_left.shape[2], HR_left.shape[3]), mode='bicubic')
                    SR_right = F.interpolate(SR_right, size=(HR_right.shape[2], HR_right.shape[3]), mode='bicubic')
                psnr_l = cal_psnr(HR_left[:, :, :, :].data.cpu().numpy(), SR_left[:, :, :, :].data.cpu().numpy())
                ssim_l = cal_ssim(HR_left[0, :, :, :].data.cpu().numpy(), SR_left[0, :, :, :].data.cpu().numpy())
                psnr_r = cal_psnr(HR_right[:, :, :, :].data.cpu().numpy(), SR_right[:, :, :, :].data.cpu().numpy())
                ssim_r = cal_ssim(HR_right[0, :, :, :].data.cpu().numpy(), SR_right[0, :, :, :].data.cpu().numpy())
                psnr_list.append(psnr_l)
                ssim_list.append(ssim_l)
                psnr_list1.append(psnr_r)
                ssim_list1.append(ssim_r)
            print('Avg_L. PSNR: {:.5f} dB, Avg. SSIM: {:.5f}'.format(np.mean(psnr_list), np.mean(ssim_list)))
            print('Avg_R. PSNR: {:.5f} dB, Avg. SSIM: {:.5f}'.format(np.mean(psnr_list1), np.mean(ssim_list1)))
            print('Avg. PSNR: {:.5f} dB, Avg. SSIM: {:.5f}'.format(np.mean(psnr_list + psnr_list1), np.mean(ssim_list + ssim_list1)))

        if idx_epoch % 1000 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print('Epoch--%4d, loss--%f ' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
            torch.save({'epoch': idx_epoch + 1, 'state_dict': net.state_dict()},
                       './log/' + cfg.model_name + '_' + str(cfg.scale_factor) + 'xSR_epoch' + str(idx_epoch + 1) + '.pth.tar')
            loss_epoch = []


def main(cfg):
    train_set = TrainSetLoader(cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=6, batch_size=cfg.batch_size, shuffle=True)
    train(train_loader, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

