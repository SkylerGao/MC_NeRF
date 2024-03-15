import argparse
import os
import logging
import torch
import random
import time
import lpips

import numpy as np 

from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

from config import Load_config
from data import Data_set 
from data import Data_loader
from model import MC_Model
from model import MC_NeRF_Loss
from model import RAdam
from model import apply_depth_colormap
from model.external.pohsun_ssim import pytorch_ssim
from utils import get_rank
from utils import Tensorboard_config


class Model_Engine():
    def __init__(self, sys_param):
        self.mode_numb = sys_param["mode"]     
        # dataset initialize 
        dataset = Data_set(sys_param)
        self.sys_param = dataset.update_system_param
        self.device = self.sys_param["device_type"]
        # epoch for different stage
        self.cam_epoch = self.sys_param["stage1_epoch"]
        self.optim_epoch = self.sys_param["stage2_epoch"]
        self.fine_tune_epoch = self.sys_param["stage3_epoch"]
        self.each_epoch = self.sys_param["epoch_squence"]
        self.total_epoch = self.sys_param["epoch_numb"]
        # loading model
        self.loader = Data_loader(dataset, self.sys_param)
        self.mc_nerf = MC_Model(self.sys_param)
        # tensorboard initialize
        self.tblogger = Tensorboard_config(self.sys_param).tblogger           
        # loss function
        self.loss_func = MC_NeRF_Loss(sys_param, self.tblogger).to(self.device)

    def forward(self):
        if self.mode_numb == 0:
            self.train_model()
        else:
            self.test_model()

    def train_model(self):
        train_loader = self.loader.dataloader["Shuffle_loader"]
        sampler_train = self.loader.sampler
        each_epoch_step = len(train_loader)
        total_step = self.total_epoch*each_epoch_step 
        cur_step = 0
        if self.sys_param["distributed"]:
            mc_nerf = torch.nn.parallel.DistributedDataParallel(self.mc_nerf.to(self.device), device_ids=[sys_param['gpu']], find_unused_parameters=True)
            self.enerf_model_without_ddp = mc_nerf.module
        else:
            mc_nerf = self.mc_nerf.to(self.device)
            self.enerf_model_without_ddp = mc_nerf
        opt_list, sched_list = self.generate_optimizer(each_epoch_step)
        # start training
        for epoch in range(self.total_epoch):
            # current epoch name
            epoch_type = self.which_stage(self.each_epoch, epoch)
            running_loss = 0
            if sys_param["distributed"]:
                sampler_train.set_epoch(epoch)   
            with tqdm(total = len(train_loader), 
                    desc='{}:{}'.format(epoch_type, epoch), 
                    bar_format='{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} {postfix}]', 
                    ncols=150) as bar:
                for step, data in enumerate(train_loader):
                    optimizer = opt_list[self.enerf_model_without_ddp.opt_idx]
                    cur_ratio = cur_step/total_step
                    optimizer.zero_grad()
                    loss_dict, intr_show, pose_show, rays_valid = mc_nerf(data, epoch, epoch_type, cur_ratio)                
                    loss_value = self.loss_func(loss_dict, epoch_type)
                    loss_value.backward()
                    optimizer.step()         
                    running_loss += loss_value.item()
                    ave_loss = running_loss/(step + 1)
                    cur_step += 1
                    sched_list[self.enerf_model_without_ddp.opt_idx].step()
                    bar.set_postfix_str('AveLoss:{:^7.9f}, LR:{:^7.5f}'.format(ave_loss, optimizer.param_groups[0]['lr']))
                    bar.update()
            self.enerf_model_without_ddp.nerf.save_model(self.mc_nerf, epoch)
            self.enerf_model_without_ddp.show_estimate_param(intr_show, pose_show, epoch, epoch_type)
            self.enerf_model_without_ddp.show_RT_est_results(epoch, epoch_type, mode='epoch')
            self.enerf_model_without_ddp.nerf.valid_train(epoch, rays_valid, epoch_type)

    @torch.no_grad()
    def test_model(self):
        test_loader = self.loader.dataloader["Squence_loader"]
        self.mc_nerf.to(self.device)
        self.mc_nerf.eval()
        res_rgbs = []
        res_invdepth = []
        img_h = sys_param["res_h"]
        img_w = sys_param["res_w"]
        img_res = img_h*img_w
        img_name_idx = 0   
        with tqdm(total = len(test_loader), 
                desc='Rendering:', 
                bar_format='{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} {postfix}]', 
                ncols=150) as bar:
            # 训练
            for step, data in enumerate(test_loader):
                gt_rgbs, img_idx = data
                # 开始前项计算
                pd_rgbs, pd_depth, pd_opacity = self.mc_nerf(img_idx.to(self.device))
                invdepth = 1/(pd_depth/pd_opacity + 1e-10) * 2
                invdepth = apply_depth_colormap(invdepth, cmap="inferno")
                res_rgbs += [pd_rgbs]
                res_invdepth += [invdepth]
                bar.update()

        rgbs_cat = torch.cat(res_rgbs, 0)
        dept_cat = torch.cat(res_invdepth, 0)
        nowtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        save_pth_pd = os.path.join(Path(sys_param["demo_render_pth"] + "_" + nowtime), Path("pred"))
        save_pth_depth = os.path.join(Path(sys_param["demo_render_pth"] + "_" + nowtime), Path("depth"))
        save_pth_gt = os.path.join(Path(sys_param["demo_render_pth"] + "_" + nowtime), Path("gt"))

        if not os.path.exists(save_pth_gt):
            os.makedirs(save_pth_gt)   
        if not os.path.exists(save_pth_pd):
            os.makedirs(save_pth_pd)              
        if not os.path.exists(save_pth_depth):
            os.makedirs(save_pth_depth)       

        PSNR_score = 0
        SSIM_score = 0
        LPIP_score = 0
        
        for i in range(0, rgbs_cat.shape[0], img_res):
            rgb_img_rays = rgbs_cat[i:i+img_res]
            dep_img_rays = dept_cat[i:i+img_res]
            gt_img = gt_rgbs.reshape(img_h, img_w, 3).cpu()
            img = rgb_img_rays.reshape(img_h, img_w, 3).cpu()
            dep = dep_img_rays.reshape(img_h, img_w, 3).cpu()
            gt_img = gt_img.permute(2, 0, 1) # (3, H, W)
            img = img.permute(2, 0, 1) # (3, H, W)
            dep = dep.permute(2, 0, 1) # (3, H, W)
            psnr = self.psnr_score(img, gt_img)
            ssim = self.ssim_score(img, gt_img)
            lpip = self.lpips_score(img, gt_img)
            gt_name = str(img_name_idx).zfill(4) + "gt" + ".png"
            img_name = str(img_name_idx).zfill(4) + ".png"
            dep_name = str(img_name_idx).zfill(4) + "depth" + ".png"
            gt_pth = os.path.join(Path(save_pth_gt), Path(gt_name))
            img_pth = os.path.join(Path(save_pth_pd), Path(img_name))
            dep_pth = os.path.join(Path(save_pth_depth), Path(dep_name))
            # 转换图片格式
            transforms.ToPILImage()(gt_img).convert("RGB").save(gt_pth)
            transforms.ToPILImage()(img).convert("RGB").save(img_pth)
            transforms.ToPILImage()(dep).convert("RGB").save(dep_pth)
            img_name_idx += 1
            PSNR_score += psnr
            SSIM_score += ssim
            LPIP_score += lpip
    
        print('Results ({})'.format(sys_param['data_name']))
        print('PSNR: {}'.format(PSNR_score/200))
        print('SSIM: {}'.format(SSIM_score/200))
        print('LPIP: {}'.format(LPIP_score/200))

        torch.cuda.empty_cache()

    # optimizer config
    def generate_optimizer(self, each_epoch_step):
        cam_stage_lr = self.sys_param["stage1_lr"]
        optim_stage_lr = self.sys_param["stage2_lr"]
        fine_tune_lr = self.sys_param["stage3_lr"]
        weight_d = self.sys_param["weight_d"]
        # camera parameters initial stage
        for key in self.enerf_model_without_ddp.named_parameters():
            if key[0].split(".")[0] == "nerf":
                key[1].requires_grad_(False)
            else:
                key[1].requires_grad_(True)
        opt_cam = RAdam(filter(lambda p: p.requires_grad, self.enerf_model_without_ddp.parameters()), lr=cam_stage_lr, eps=1e-8, weight_decay=weight_d)
        gamma_cam = (0.005/cam_stage_lr)**(1./(each_epoch_step*self.cam_epoch))
        opt_cam_sched = torch.optim.lr_scheduler.ExponentialLR(opt_cam, gamma_cam)
        # global optimization stage
        for key in self.enerf_model_without_ddp.named_parameters():
            key[1].requires_grad_(True)
        opt_global = RAdam(filter(lambda p: p.requires_grad, self.enerf_model_without_ddp.parameters()), lr=optim_stage_lr, eps=1e-8, weight_decay=weight_d)
        gamma_global = (optim_stage_lr/optim_stage_lr)**(1./(each_epoch_step*self.optim_epoch))
        opt_global_sched = torch.optim.lr_scheduler.ExponentialLR(opt_global, gamma_global)
        # fine tuning stage
        for key in self.enerf_model_without_ddp.named_parameters():
            key[1].requires_grad_(True)
        self.enerf_model_without_ddp.weights_pose.requires_grad_(False)
        opt_fine_tune = RAdam(filter(lambda p: p.requires_grad, self.enerf_model_without_ddp.parameters()), lr=fine_tune_lr, eps=1e-8, weight_decay=weight_d)
        gamma_fine_tune = (fine_tune_lr/fine_tune_lr)**(1./(each_epoch_step*self.fine_tune_epoch))
        opt_fine_sched = torch.optim.lr_scheduler.ExponentialLR(opt_fine_tune, gamma_fine_tune)
        # activate all weights before training
        for key in self.enerf_model_without_ddp.named_parameters():
            key[1].requires_grad_(True)
            
        return [opt_cam, opt_global, opt_fine_tune], [opt_cam_sched, opt_global_sched, opt_fine_sched]

    # get current stage info
    def which_stage(self, epoch_list, cur_epoch):
        epoch_name = ['CAM_PARAM_EPOCH', 'GLOBAL_OPTIM_EPOCH', 'FINE_TUNE_EPOCH']
        name_idx = torch.arange(0, len(epoch_name))
        epoch_list = torch.cumsum(epoch_list, 0)
        for idx, epoch in enumerate(epoch_list):
            if cur_epoch in range(epoch):
                cur_idx = name_idx[idx]
                return epoch_name[cur_idx]

    # get PSNR score
    def psnr_score(self, image_pred, image_gt, valid_mask=None, reduction='mean'):
        def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
            value = (image_pred-image_gt)**2
            if valid_mask is not None:
                value = value[valid_mask]
            if reduction == 'mean':
                return torch.mean(value)
            return value
        return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

    # get SSIM score
    def ssim_score(self, image_pred, image_gt):
        image_pred = image_pred.unsqueeze(0)
        image_gt = image_gt.unsqueeze(0)
        ssim = pytorch_ssim.ssim(image_pred, image_gt).item()
        return ssim

    # get lpips score
    def lpips_score(self, image_pred, image_gt):
        lpips_loss = lpips.LPIPS(net="alex")
        lpips_value = lpips_loss(image_pred*2-1, image_gt*2-1).item()
        return lpips_value


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    # config file path
    parser.add_argument('--config', type=str, default="./config",
                        help='root path of config file')
    # root path for data
    parser.add_argument('--root_data', type=str, default='./data/dataset_Ball',
                        help='root path of data')
    parser.add_argument('--data_name', type=str, default='Ball_Computer',
                        help='name of data')
    # work mode: train for train and valid, demo for test
    parser.add_argument('--demo', action='store_true',
                        help='nerf rendering forward with test mode')
    parser.add_argument('--train', action='store_true',
                        help='train mode')
    # save log file or not
    parser.add_argument('--log', action='store_true',
                        help='save log information to log.txt file')
    # GPU number, which start, available in muti-GPU training
    parser.add_argument('--start_device', type=int, default=0,
                        help='start training device for distributed mode')
    # active Tensorboard or not
    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard tools to show training results')
    args = parser.parse_args()

    config_info = Load_config(args)
    sys_param = config_info.system_info
    logging.info("System Parameters:\n {} \n".format(sys_param))
    # fix seed for reproduction
    seed = sys_param["seed"] + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # model engine
    model = Model_Engine(sys_param)
    model.forward()