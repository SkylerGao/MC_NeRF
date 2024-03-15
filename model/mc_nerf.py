import torch
import logging
import os
import time
import cv2
import json
import lpips

import numpy as np
import torch.nn as nn
import torch.distributed as dist
import prettytable as pt
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision import transforms
from mpl_toolkits.mplot3d import Axes3D

from .net_block import SinCosEmbedding
from .net_block import CorseFine_NeRF
from .net_utils import get_rank
from .external.pohsun_ssim import pytorch_ssim

class MC_Model(nn.Module):
    def __init__(self, sys_param):
        logging.info('Creating MC-NeRF Model...')
        super(MC_Model, self).__init__()
        self.sys_param = sys_param
        self.mode = self.sys_param["mode"]
        self.device = self.sys_param["device_type"]
        self.batch = self.sys_param["batch"]
        self.bound_min = self.sys_param["boader_min"]
        self.bound_max = self.sys_param["boader_max"]
        self.intr = self.sys_param["intr_mat"]
        self.intr_inv = self.sys_param["intr_mat_inv"]
        self.intr_train, self.intr_test, self.intr_val = self.intr[0], self.intr[1], self.intr[2]
        self.intr_train_inv, self.intr_test_inv, self.intr_val_inv = self.intr_inv[0], self.intr_inv[1], self.intr_inv[2]
        self.gt_pose = self.sys_param["gt_pose"].to(self.device)
        self.test_pose = self.sys_param["test_pose"].to(self.device)
        self.valid_pose = self.sys_param["valid_pose"].to(self.device)
        self.valid_rgbs = self.sys_param["valid_rgbs"].to(self.device)
        self.img_h = self.sys_param["data_img_h"]
        self.img_w = self.sys_param["data_img_w"]
        self.train_img_pth = self.sys_param["demo_render_pth"]
        self.data_name = self.sys_param["data_name"]
        self.data_numb = self.sys_param["data_numb"]
        self.train_numb, self.test_numb, self.val_numb = self.data_numb[0], self.data_numb[1], self.data_numb[2]
        self.train_json_pth = sys_param["train_json_file"]
        self.register_parameters()
        self.nerf = NeRF_Model(self.sys_param).to(self.device)
        self.table = pt.PrettyTable(['EPOCH', 'LOSS_FX', 'LOSS_FY', 'LOSS_UX', 'LOSS_UY', 'LOSS_K', 'LOSS_R', 'LOSS_T'])
        self.count_rays = 0
        self.init_show_figure(show_info=False)
        self.opt_idx = 0
        self.last_epoch_type = 0
        self.wait_reset = 0

    def forward(self, *args):
        if self.sys_param["mode"] == 0:
            loss_dict = {}
            gt_rgbs, img_id, intr_wpts, intr_pts,\
            extr_wpts, extr_pts, epoch, epoch_type, cur_ratio = self.data2device(*args)
            # camera parameters initial stage
            if epoch_type == "CAM_PARAM_EPOCH":
                self.nerf.emmbedding_xyz.barf_mode = False
                self.intr_adj, self.pose_adj, self.calib_pose_adj = self.add_weights2param(intr=True, extr=True, calib_extr=True)
                reproj_pts_intr = self.get_reproject_pixels(intr_wpts, self.intr_adj, self.calib_pose_adj)
                reproj_pts_extr = self.get_reproject_pixels(extr_wpts, self.intr_adj, self.pose_adj)
                loss_dict["intr"] = [reproj_pts_intr, intr_pts]
                loss_dict["extr"] = [reproj_pts_extr, extr_pts]
                self.opt_idx = 0
            # global optimization stage
            elif epoch_type == "GLOBAL_OPTIM_EPOCH":
                self.nerf.emmbedding_xyz.barf_mode = True
                self.intr_adj, self.pose_adj, self.calib_pose_adj = self.add_weights2param(intr=True, extr=True, calib_extr=True)
                reproj_pts_intr = self.get_reproject_pixels(intr_wpts, self.intr_adj, self.calib_pose_adj)
                rays_d, rays_o = self.get_rays(self.pose_adj, img_id, self.inverse_intrinsic(self.intr_adj))
                sample_rays_d, sample_rays_o, rand_idx = self.generate_rand_rays(rays_d, rays_o, rand=True)
                rgbs_c, rgbs_f = self.nerf(sample_rays_d, sample_rays_o, epoch, cur_ratio)
                gt_rgbs = gt_rgbs.reshape(-1, 3)[rand_idx]
                loss_dict["intr"] = [reproj_pts_intr, intr_pts]
                loss_dict["rgb"] = [rgbs_c, rgbs_f, gt_rgbs]
                self.opt_idx = 1
            # fine-tuning stage
            else:
                self.nerf.emmbedding_xyz.barf_mode = False
                self.intr_adj, self.pose_adj, self.calib_pose_adj = self.add_weights2param(intr=True, extr=False, calib_extr=True)
                reproj_pts_intr = self.get_reproject_pixels(intr_wpts, self.intr_adj, self.calib_pose_adj)
                rays_d, rays_o = self.get_rays(self.pose_adj, img_id, self.inverse_intrinsic(self.intr_adj))
                sample_rays_d, sample_rays_o, rand_idx = self.generate_rand_rays(rays_d, rays_o, rand=True)
                rgbs_c, rgbs_f = self.nerf(sample_rays_d, sample_rays_o, epoch, 1)
                gt_rgbs = gt_rgbs.reshape(-1, 3)[rand_idx]
                loss_dict["intr"] = [reproj_pts_intr, intr_pts]
                loss_dict["rgb"] = [rgbs_c, rgbs_f, gt_rgbs]
                self.opt_idx = 2             
            # generate valid data
            rays_dv, rays_ov = self.get_rays(self.valid_pose, img_id, self.intr_val_inv.to(self.device))
            rgbs_v = self.valid_rgbs[img_id]
            rays_valid = [rays_dv.detach(), rays_ov.detach(), rgbs_v.detach()]

            intr_show = [self.intr_train.to(self.device).detach(), self.intr_adj.detach()]
            pose_show = [self.gt_pose.to(self.device).detach(), self.pose_adj.detach()]
            self.last_epoch_type = epoch_type
            
            return loss_dict, intr_show, pose_show, rays_valid
        else:
            img_id = args
            rgbs = []
            depth = []
            opacity = []
            rays_d, rays_o = self.get_rays(self.test_pose, img_id, self.intr_test_inv.to(self.device))  
            for ii in range(0, rays_d.shape[0], self.batch):
                cur_rays_d = rays_d[ii:ii+self.batch]
                cur_rays_o = rays_o[ii:ii+self.batch]
                pd_rgbs, pd_depth, pd_opacity = self.nerf(cur_rays_d, cur_rays_o)
                rgbs += [pd_rgbs.detach().cpu()]
                depth += [pd_depth.detach().cpu()]
                opacity += [pd_opacity.detach().cpu()]
            rgbs = torch.cat(rgbs, 0)
            depth = torch.cat(depth, 0)
            opacity = torch.cat(opacity, 0)
            return rgbs, depth, opacity

    def get_rays(self, pose, img_id, intr_inv):
        select_pose = pose[img_id]
        with torch.no_grad():
            y_range = torch.arange(self.img_h, dtype=torch.float32, device=self.device).add_(0.5)
            x_range = torch.arange(self.img_w, dtype=torch.float32, device=self.device).add_(0.5)
            Y,X = torch.meshgrid(y_range,x_range, indexing='ij') # [H,W]
            xy_grid = torch.stack([X,Y],dim=-1).reshape(-1,2) # [HW,2]
        xy_grid = xy_grid.unsqueeze(0) # [1,HW,2]
        pix_cord = self.pix2hom(xy_grid)
        cam_cord = self.pix2cam(pix_cord, intr_inv[img_id])
        cam_orig = torch.zeros_like(cam_cord)
        cam_cord_hom = self.cam2hom(cam_cord)
        cam_orig_hom = self.cam2hom(cam_orig)
        world_cord = self.cam2world(cam_cord_hom, select_pose)
        rays_o = self.cam2world(cam_orig_hom, select_pose)
        rays_d = world_cord - rays_o
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        rays_d = rays_d.reshape(-1, 3)
        rays_o = rays_o.reshape(-1, 3)
        
        return rays_d, rays_o 

    def get_reproject_pixels(self, tag_wpts, intr_adj, pose_adj):
        world_pts = self.world2hom(tag_wpts)
        proj_cam_pts = self.world2cam(world_pts, pose_adj.unsqueeze(0))
        proj_pts = self.cam2pix(proj_cam_pts, intr_adj.unsqueeze(0))

        return proj_pts

    # transform parameters to learnable parameters
    def add_weights2param(self, intr=True, extr=True, calib_extr=False):
        if intr:
            intr_adj = self.add_weights2intr(self.img_h, self.img_w)
        else:
            intr_adj = self.add_weights2intr(self.img_h, self.img_w, adj=False)
        if extr:
            pose_adj = self.add_weights2pose()
        else:
            pose_adj = self.add_weights2pose(adj=False)
        if calib_extr:
            calib_pose_adj = self.add_weights2calib_pose()
        else:
            calib_pose_adj = self.add_weights2calib_pose(adj=False)
        
        return intr_adj, pose_adj, calib_pose_adj

    def add_weights2intr(self, img_h, img_w, adj=True):        
        intr_init = torch.tensor([[img_w, 0, img_w/2],
                                  [0, img_w, img_h/2],
                                  [0,       0,       1]], device=self.device).expand(self.train_numb, 3, 3)
        intr_adj = intr_init.clone()
        if adj:
            intr_adj[:, 0, 0] = torch.abs(intr_init[:, 0, 0]*self.weights_fx.requires_grad_(True))
            intr_adj[:, 1, 1] = torch.abs(intr_init[:, 1, 1]*self.weights_fy.requires_grad_(True))
            intr_adj[:, 0, 2] = torch.abs(intr_init[:, 0, 2]*self.weights_ux.requires_grad_(True))
            intr_adj[:, 1, 2] = torch.abs(intr_init[:, 1, 2]*self.weights_uy.requires_grad_(True))    
        else:
            intr_adj[:, 0, 0] = torch.abs(intr_init[:, 0, 0]*self.weights_fx.requires_grad_(False))
            intr_adj[:, 1, 1] = torch.abs(intr_init[:, 1, 1]*self.weights_fy.requires_grad_(False))
            intr_adj[:, 0, 2] = torch.abs(intr_init[:, 0, 2]*self.weights_ux.requires_grad_(False))
            intr_adj[:, 1, 2] = torch.abs(intr_init[:, 1, 2]*self.weights_uy.requires_grad_(False))              
        return intr_adj

    def add_weights2pose(self, adj=True):
        if adj:
            weights_RT = self.se3_to_SE3(self.weights_pose.requires_grad_(True))
        else:
            weights_RT = self.se3_to_SE3(self.weights_pose.requires_grad_(False))        

        return weights_RT

    def add_weights2calib_pose(self, adj=True):
        if adj:
            weights_RT = self.se3_to_SE3(self.weights_pose_intr.requires_grad_(True))
        else:
            weights_RT = self.se3_to_SE3(self.weights_pose_intr.requires_grad_(False))        

        return weights_RT

    def inverse_intrinsic(self, intr_mats):
        intr_inv_list = []
        for intr in intr_mats:
            intr_inv = intr.inverse()
            intr_inv_list += [intr_inv]
        intr_inv = torch.stack(intr_inv_list, 0)
        return intr_inv
       
    # [Batch, HW, 2]->[Batch, HW, 3]
    def pix2hom(self, pixel_cord):
        X_hom = torch.cat([pixel_cord, torch.ones_like(pixel_cord[...,:1])], dim=-1)
        return X_hom

    # [Batch, HW, 3]->[Batch, HW, 4]
    def cam2hom(self, cam_cord):
        X_hom = torch.cat([cam_cord, torch.ones_like(cam_cord[...,:1])], dim=-1)
        return X_hom
    
    # [Batch, HW, 3]->[Batch, HW, 4]
    def world2hom(self, world_cord):
        X_hom = torch.cat([world_cord, torch.ones_like(world_cord[...,:1])], dim=-1)
        return X_hom        
          
    # pix_cord: [batch, ..., 3]
    # intr_inv_mat: [batch, ..., 3, 3]
    def pix2cam(self, pix_cord, intr_inv_mat):
        intr_inv_mat = intr_inv_mat.transpose(-2, -1)
        cam_cord = pix_cord @ intr_inv_mat
        return cam_cord

    # cam_cord: [batch, ..., 4]
    # intr_mat: [batch, ..., 3, 3]    
    def cam2pix(self, cam_cord, intr_mat):
        hom_intr_mat = torch.cat([intr_mat, torch.zeros_like(intr_mat[...,:1])], dim=-1)
        pix_cord = hom_intr_mat @ cam_cord
        pix_cord = pix_cord[...,:2,:]/pix_cord[...,2:,:]
        pix_cord = pix_cord.transpose(-2, -1)
        return pix_cord

    # cam_cord: [batch, ..., 4]
    # pose: [batch, ..., 4]
    def cam2world(self, cam_cord, pose):
        pose_R = pose[..., :3]
        pose_T = pose[..., 3:]
        # 正交矩阵，转置等于逆矩阵
        pose_R_inv = pose_R.transpose(-2, -1)
        pose_T_inv = (-pose_R_inv @ pose_T)
        # [batch, 3, 4]
        pose_inv = torch.cat([pose_R_inv, pose_T_inv], -1)
        # [batch, HW, 3]
        world_cord = cam_cord @ pose_inv.transpose(-2, -1)

        return world_cord
    
    # world_cord: [batch, ..., 4]
    # pose: [batch, ..., 3, 4]
    def world2cam(self, world_cord, pose):
        shape = pose.shape
        supply_pose = torch.tensor([0, 0, 0, 1], device=self.device)
        supply_pose = supply_pose.expand(shape)[...,:1,:]
        hom_pose = torch.cat([pose, supply_pose], dim=-2)
        cam_cord = hom_pose @ world_cord.transpose(-2, -1)

        return cam_cord

    def se3_to_SE3(self, wu): # [...,3]
        w,u = wu.split([3,3],dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3, device=self.device)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I+A*wx+B*wx@wx
        V = I+B*wx+C*wx@wx
        Rt = torch.cat([R,(V@u[...,None])],dim=-1)

        return Rt

    def skew_symmetric(self, w):
        w0,w1,w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                          torch.stack([w2,O,-w0],dim=-1),
                          torch.stack([-w1,w0,O],dim=-1)],dim=-2)
        return wx

    def taylor_A(self,x,nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i>0: denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    
    def taylor_B(self,x,nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    
    def taylor_C(self,x,nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2)*(2*i+3)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

    def compose_param2pose(self, param, pose):
        R_a,t_a = param[...,:3], param[...,3:]
        R_b,t_b = pose[...,:3], pose[...,3:]
        R_new = R_b @ R_a
        t_new = (R_b @ t_a + t_b)
        pose_new = torch.cat([R_new, t_new], -1)

        return pose_new
    
    def generate_rand_rays(self, rays_d, rays_o, rand=True):
        if rand:
            rand_idx = torch.randperm(rays_d.shape[0], device = self.device)[:self.batch]
        else:
            slides = torch.arange(0, rays_d.shape[0], self.batch, device=self.device)
            if self.count_rays == len(slides):
                self.count_rays = 0
            if slides[self.count_rays] + self.batch > rays_d.shape[0]:
                end_idx = rays_d.shape[0]
            else:
                end_idx = slides[self.count_rays] + self.batch
            rand_idx = torch.arange(slides[self.count_rays], end_idx, device=self.device)
   
        sample_rays_d = rays_d[rand_idx]
        sample_rays_o = rays_o[rand_idx]
        
        self.count_rays += 1
 
        return sample_rays_d, sample_rays_o, rand_idx

    def register_parameters(self):
        self.register_parameter(
            name="weights_pose",
            param=nn.Parameter(torch.ones([self.train_numb, 6],
                                            device=self.device), requires_grad=True))
        self.register_parameter(
            name="weights_pose_intr",
            param=nn.Parameter(torch.ones([self.train_numb, 6],
                                            device=self.device), requires_grad=True))      
        self.register_parameter(
            name="weights_ux",
            param=nn.Parameter(torch.ones([self.train_numb], 
                                            device=self.device), requires_grad=True))
        self.register_parameter(
            name="weights_uy",
            param=nn.Parameter(torch.ones([self.train_numb], 
                                            device=self.device), requires_grad=True))
        self.register_parameter(
            name="weights_fx",
            param=nn.Parameter(torch.ones([self.train_numb], 
                                            device=self.device), requires_grad=True))
        self.register_parameter(
            name="weights_fy",
            param=nn.Parameter(torch.ones([self.train_numb], 
                                            device=self.device), requires_grad=True))

    # data to GPU
    def data2device(self, *args):
        gt_rgbs, img_id, intr_wpts, intr_pts, extr_wpts, extr_pts = args[0]
        epoch = args[1]
        epoch_type = args[2]
        cur_ratio = args[3]
        gt_rgbs = gt_rgbs.to(self.device) 
        img_id = img_id.to(self.device)
        intr_wpts = intr_wpts.to(self.device)
        intr_pts = intr_pts.to(self.device)
        extr_wpts = extr_wpts.to(self.device)
        extr_pts = extr_pts.to(self.device)

        return gt_rgbs, img_id, intr_wpts, intr_pts, extr_wpts, extr_pts, epoch, epoch_type, cur_ratio

    def show_estimate_param(self, intr_show, pose_show, epoch, epoch_type):
        intr_loss = torch.abs(intr_show[0] - intr_show[1])
        pose_loss = torch.abs(pose_show[0] - pose_show[1])
        ave_loss_fx = intr_loss[:,0,0].mean()
        ave_loss_fy = intr_loss[:,1,1].mean()
        ave_loss_ux  = intr_loss[:,0,2].mean()
        ave_loss_uy  = intr_loss[:,1,2].mean()
        ave_loss_K = intr_loss.mean()

        ave_loss_R = pose_loss[:,:3,:3].mean()
        ave_loss_T = pose_loss[:,:3,3:].mean()
        self.table.add_row([int(epoch), round(float(ave_loss_fx), 4),\
                                        round(float(ave_loss_fy), 4),\
                                        round(float(ave_loss_ux), 4),\
                                        round(float(ave_loss_uy), 4),\
                                        round(float(ave_loss_K), 4),\
                                        round(float(ave_loss_R), 4),\
                                        round(float(ave_loss_T), 4)])

        print(self.table)

    def init_show_figure(self, show_info=True):
        self.all_fig = plt.figure(figsize=(4,4))
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams['mathtext.default'] = 'regular'
        self.ax = Axes3D(self.all_fig, auto_add_to_figure=False)
        self.all_fig.add_axes(self.ax)
        self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            
        if show_info:
            self.ax.set_xlabel("X Axis")
            self.ax.set_ylabel("Y Axis")
            self.ax.set_zlabel("Z Axis")
            self.ax.set_xlim(-3.5, 3.5)
            self.ax.set_ylim(-3.5, 3.5)
            self.ax.set_zlim(-3.5, 3.5)
        else:
            self.ax.grid(False)
            self.ax.axis(False)

        plt.ion()
        plt.gca().set_box_aspect((1, 1, 1))

    def origin_pose_transform(self, pose):
        pose_R_new_inv = pose[..., :3] 
        pose_T_new_inv = pose[..., 3:] 
        pose_R_new = pose_R_new_inv.transpose(-2, -1)
        pose_T_new = -pose_R_new @ pose_T_new_inv
        pose_flip_R_inv = torch.diag(torch.tensor([1.0,-1.0,-1.0], device=self.device)).T.expand(pose.shape[0], 3, 3)
        pose_flip_T = torch.zeros([3, 1], device=self.device).expand(pose.shape[0], 3, 1)

        pose_R = pose_R_new @ pose_flip_R_inv
        pose_T = pose_T_new - pose_R @ pose_flip_T
        pose_ori = torch.cat([pose_R, pose_T], -1)

        return pose_ori     
    
    def show_RT_est_results(self, epoch, epoch_type, mode='epoch', show_info=True):
        if epoch_type in ['INTR_EPOCH']:
            return 0
        
        # 判断一下路径是否存在
        save_path = os.path.join(Path(self.train_img_pth), Path(self.data_name), Path("cam_pose"))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if mode == 'epoch':
            plt.cla()
            color_gt = (0.7,0.2,0.7)
            color_pd = (0,0.6,0.7)
            # [84, 3, 4]
            gt_ori_pose = self.origin_pose_transform(self.gt_pose.detach())
            pd_ori_pose = self.origin_pose_transform(self.pose_adj.detach())        
            self.draw_camera_shape(gt_ori_pose, self.intr_train, color_gt, cam_size=0.3)
            self.draw_camera_shape(pd_ori_pose, self.intr_adj.detach(), color_pd, cam_size=0.3)

            if show_info:
                self.ax.set_xlim(-3.5, 3.5)
                self.ax.set_ylim(-3.5, 3.5)
                self.ax.set_zlim(-3.5, 3.5)
            else:
                self.ax.grid(False)
                self.ax.axis(False)
                self.ax.txt

            file_path = os.path.join(Path(save_path), Path("epoch_"+ str(epoch) + ".png"))
            plt.savefig(file_path)
        else:
            if epoch % 100 == 0:
                plt.cla()
                color_gt = (0.7,0.2,0.7)
                color_pd = (0,0.6,0.7)
                gt_ori_pose = self.origin_pose_transform(self.gt_pose.detach())
                pd_ori_pose = self.origin_pose_transform(self.pose_adj.detach())        
                self.draw_camera_shape(gt_ori_pose, self.intr_train, color_gt, cam_size=0.3)
                self.draw_camera_shape(pd_ori_pose, self.intr_adj.detach(), color_pd, cam_size=0.3)

                if show_info:
                    self.ax.set_xlim(-3.5, 3.5)
                    self.ax.set_ylim(-3.5, 3.5)
                    self.ax.set_zlim(-3.5, 3.5)
                else:
                    self.ax.grid(False)
                    self.ax.axis(False)
                                        
                file_path = os.path.join(Path(save_path), Path("step_"+ str(epoch) + ".png"))
                plt.savefig(file_path, dpi=500)

    def draw_camera_shape(self, extr_mat, intr_mat, color, cam_size=0.25):
        # extr_mat: [84, 3, 4]
        # intr_mat: [84, 3, 3]
        cam_line = cam_size
        focal = intr_mat[:,0,0]*cam_line/self.img_w
        cam_pts_1 = torch.stack([-torch.ones_like(focal)*cam_line/2,
                                 -torch.ones_like(focal)*cam_line/2,
                                 -focal], -1)[:,None,:].to(extr_mat.device)
        cam_pts_2 = torch.stack([-torch.ones_like(focal)*cam_line/2,
                                  torch.ones_like(focal)*cam_line/2,
                                 -focal], -1)[:,None,:].to(extr_mat.device)
        cam_pts_3 = torch.stack([ torch.ones_like(focal)*cam_line/2,
                                  torch.ones_like(focal)*cam_line/2,
                                  -focal], -1)[:,None,:].to(extr_mat.device)
        cam_pts_4 = torch.stack([ torch.ones_like(focal)*cam_line/2,
                                 -torch.ones_like(focal)*cam_line/2,
                                 -focal], -1)[:,None,:].to(extr_mat.device)
        cam_pts_1 = cam_pts_1 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
        cam_pts_2 = cam_pts_2 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
        cam_pts_3 = cam_pts_3 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
        cam_pts_4 = cam_pts_4 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
        cam_pts = torch.cat([cam_pts_1, cam_pts_2, cam_pts_3, cam_pts_4, cam_pts_1], dim=-2)
        for i in range(4):
            # [84, 2, 3]
            cur_line_pts = torch.stack([cam_pts[:,i,:], cam_pts[:,i+1,:]], dim=-2).to('cpu')
            for each_cam in cur_line_pts:
                self.ax.plot(each_cam[:,0],each_cam[:,1],each_cam[:,2],color=color,linewidth=0.5)
        extr_T = extr_mat[:, :3, 3]
        for i in range(4):
            # [84, 2, 3]
            cur_line_pts = torch.stack([extr_T, cam_pts[:,i,:]], dim=-2).to('cpu')
            for each_cam in cur_line_pts:
                self.ax.plot(each_cam[:,0],each_cam[:,1],each_cam[:,2],color=color,linewidth=0.5)
        extr_T = extr_T.to('cpu')

        self.ax.scatter(extr_T[:,0],extr_T[:,1],extr_T[:,2],color=color,s=5)

    def listify_matrix(self, matrix):
        matrix_list = []
        for row in matrix:
            matrix_list.append(list(row))
        return matrix_list
    
# NeRF渲染模型
class NeRF_Model(nn.Module):
    def __init__(self, sys_param):
        logging.info('Creating NeRF Model...')
        super(NeRF_Model, self).__init__()
        self.sys_param = sys_param
        self.mode = self.sys_param["mode"]
        self.device = self.sys_param["device_type"]
        self.near = self.sys_param["near"]
        self.far = self.sys_param["far"]
        self.samples_c = self.sys_param["samples"]
        self.sample_scale = self.sys_param["scale"]
        self.samples_f = self.samples_c * self.sample_scale
        self.dim_sh = 3 * (self.sys_param["MLP_deg"] + 1)**2
        self.white_back = self.sys_param["white_back"]
        self.weights_pth = self.sys_param['root_weight']
        self.train_img_pth = self.sys_param["demo_render_pth"]
        self.batch_test = self.sys_param["batch"]
        self.xyz_min = self.sys_param["boader_min"]
        self.xyz_max = self.sys_param["boader_max"]
        self.xyz_scope = self.xyz_max - self.xyz_min
        self.grid_nerf = self.sys_param["grid_nerf"]
        self.sigma_init = self.sys_param["sigma_init"]
        self.sigma_default = self.sys_param["sigma_default"]
        self.warmup_epoch = self.sys_param["warmup_epoch"]
        self.weight_thresh = self.sys_param['sample_weight_thresh']
        self.render_h = self.sys_param["res_h"]
        self.render_w = self.sys_param["res_w"]
        self.z_vals_c = torch.linspace(self.near, self.far, self.samples_c, device=self.device)
        self.z_vals_f = torch.linspace(self.near, self.far, self.samples_f, device=self.device)
        self.global_step = 0
        self.emmbedding_xyz = SinCosEmbedding(self.sys_param)
        self.nerf_coarse = CorseFine_NeRF(self.sys_param, type="coarse")
        self.nerf_fine   = CorseFine_NeRF(self.sys_param, type="fine")
        self.data_name = self.sys_param["data_name"]
        if self.mode != 0:
            self.nerf_ckpt_name = self.sys_param['demo_ckpt']
            self.nerf_ckpt_file = torch.load(Path(self.nerf_ckpt_name), map_location = self.device)
            self.nerf_ckpt_coarse = self.rewrite_nerf_ckpt(self.nerf_ckpt_file, coarse=True)
            self.nerf_ckpt_fine = self.rewrite_nerf_ckpt(self.nerf_ckpt_file)
            self.nerf_coarse.load_state_dict(self.nerf_ckpt_coarse) 
            self.nerf_fine.load_state_dict(self.nerf_ckpt_fine)
            logging.info("Loading weights:{}".format(self.nerf_ckpt_name))
            
    def forward(self, *args):
        self.global_step += 1
        if self.mode == 0:
            rays_d, rays_o, cur_epoch, step_r = args
            rgb_c, rgb_f = self.render_rays_train(rays_d, rays_o, cur_epoch, step_r, only_coarse=False)
            return rgb_c, rgb_f
        else:
            rays_d, rays_o = args
            results = self.render_rays_test(rays_d, rays_o, self.nerf_coarse, self.nerf_fine)

            return results

    def render_rays_train(self, rays_d, rays_o, cur_epoch, step_r, only_coarse = False):
        z_vals_samples_c = self.z_vals_c.clone().expand(rays_d.shape[0], -1)
        delta_z_vals_init = torch.empty(rays_d.shape[0], 1, device=self.device).uniform_(0.0, (self.far - self.near)/self.samples_c)
        z_vals_samples_c = z_vals_samples_c + delta_z_vals_init
        xyz_samples_c = rays_o.unsqueeze(1) + rays_d.unsqueeze(1)*z_vals_samples_c.unsqueeze(2)
        # [chunk,3]/[chunk, 256]
        rgb_coarse, sigmas_coarse, xyz_idx, depth_coarse, _ = \
        self.inference(self.nerf_coarse, 
                        self.emmbedding_xyz,
                        step_r, 
                        xyz_samples_c,
                        rays_d,
                        z_vals_samples_c)
        if only_coarse:
            return rgb_coarse, None, depth_coarse
        with torch.no_grad():
            # [self.batch, 127]
            deltas_coarse = z_vals_samples_c[:, 1:] - z_vals_samples_c[:, :-1]
            delta_inf = 1e10 * torch.ones_like(deltas_coarse[:, :1]) 
            # [self.batch, 128]
            deltas_coarse = torch.cat([deltas_coarse, delta_inf], -1)
            weights_coarse = self.sigma2weights(deltas_coarse, sigmas_coarse)
            # [self.batch, 128]
            weights_coarse = weights_coarse.detach()
        # [X, 2]
        idx_render = torch.nonzero(weights_coarse >= min(self.weight_thresh, weights_coarse.max().item()))
        # [X, 5, 2]
        idx_render = idx_render.unsqueeze(1).expand(-1, self.sample_scale, -1)
        # [X, 5, 2]
        idx_render_fine = idx_render.clone()
        idx_render_fine[..., 1] = idx_render[..., 1] * self.sample_scale + (torch.arange(self.sample_scale, device=self.device)).reshape(1, self.sample_scale)
        idx_render_fine = idx_render_fine.reshape(-1, 2)
        if idx_render_fine.shape[0] > rays_d.shape[0] * 128:
            indices = torch.randperm(idx_render_fine.shape[0])[:rays_d.shape[0] * 128]
            idx_render_fine = idx_render_fine[indices]
        z_vals_samples_f = self.z_vals_f.clone().expand(rays_d.shape[0], -1)
        z_vals_samples_f = z_vals_samples_f + delta_z_vals_init
        xyz_samples_f = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_samples_f.unsqueeze(2)
        # [self.batch,3]/[self.batch, 256]
        rgb_fine, sigmas_fine, xyz_idx, depth_fine, _ = \
        self.inference(self.nerf_fine, 
                        self.emmbedding_xyz,
                        step_r, 
                        xyz_samples_f,
                        rays_d,
                        z_vals_samples_f,
                        idx_render=idx_render_fine,
                        coarse=False)
        return rgb_coarse, rgb_fine

    def render_rays_test(self, rays_d, rays_o, model_coarse, model_fine):
        step_r = 1
        z_vals_samples_c = self.z_vals_c.clone().expand(rays_d.shape[0], -1)
        xyz_samples_c = rays_o.unsqueeze(1) + rays_d.unsqueeze(1)*z_vals_samples_c.unsqueeze(2)      
        rgb_coarse, sigmas_coarse, _, depth_coarse, opacity_coarse = self.inference(model_coarse, 
                                                                                    self.emmbedding_xyz,
                                                                                    step_r, 
                                                                                    xyz_samples_c,
                                                                                    rays_d,
                                                                                    z_vals_samples_c)  
        deltas_samples = z_vals_samples_c[:, 1:] - z_vals_samples_c[:, :-1]    
        delta_inf_samples = 1e10 * torch.ones_like(deltas_samples[:, :1])
        # [self.batch, 128]
        deltas = torch.cat([deltas_samples, delta_inf_samples], -1)  
        weights = self.sigma2weights(deltas, sigmas_coarse)
        idx_render = torch.nonzero(weights >= min(self.weight_thresh, weights.max().item()))
        idx_render = idx_render.unsqueeze(1).expand(-1, self.sample_scale, -1)
        idx_render_fine = idx_render.clone()
        idx_render_fine[..., 1] = idx_render[..., 1] * self.sample_scale + (torch.arange(self.sample_scale, device=self.device)).reshape(1, self.sample_scale)
        idx_render_fine = idx_render_fine.reshape(-1, 2)
        z_vals_samples_f = self.z_vals_f.clone().expand(rays_d.shape[0], -1)
        xyz_samples_f = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_samples_f.unsqueeze(2)        
        # [self.batch,3]/[self.batch, 256]
        rgb_fine, _, _, depth_fine, opacity_fine = self.inference(model_fine, 
                                                    self.emmbedding_xyz,
                                                    step_r, 
                                                    xyz_samples_f,
                                                    rays_d,
                                                    z_vals_samples_f,
                                                    idx_render=idx_render_fine,
                                                    coarse=False)
        # rgb_final = rgb_coarse + rgb_fine            
        return rgb_fine, depth_fine, opacity_fine
  
    def inference(self, model, embedding_xyz, step_r, xyz, rays_d, z_vals, idx_render=None, coarse=True):
        if coarse:
            sample_numb = self.samples_c
        else:
            sample_numb = self.samples_f
        batch_numb = rays_d.shape[0]
        view_dir = rays_d.unsqueeze(1).expand(-1, sample_numb, -1)
        if idx_render != None:
            view_dir = view_dir[idx_render[:, 0], idx_render[:, 1]]  
            xyz = xyz[idx_render[:, 0], idx_render[:, 1]]
            out_rgb = torch.full((batch_numb, sample_numb, 3), 1.0, device=self.device)
            out_sigma = torch.full((batch_numb, sample_numb, 1), self.sigma_default, device=self.device)
            out_defaults = torch.cat([out_sigma, out_rgb], dim=2)
        else:
            xyz = xyz.reshape(-1, 3)
            view_dir = view_dir.reshape(-1, 3)
        input_encode_xyz = embedding_xyz(xyz, step_r)
        nerf_model_out = model(input_encode_xyz, view_dir)
        if idx_render != None:
            out_defaults[idx_render[:, 0], idx_render[:, 1]] = nerf_model_out
        else:
            # 需要修改
            out_defaults = nerf_model_out.reshape(batch_numb, sample_numb, 4)
        sigmas, rgbs = torch.split(out_defaults, (1, 3), dim=-1)  
        sigmas = sigmas.squeeze(-1)
        rays_length = rays_d.norm(dim = -1, keepdim = True)
        deltas = z_vals[:, 1:] - z_vals[:, :-1]
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) 
        deltas = torch.cat([deltas, delta_inf], -1)        
        dist_samples = deltas * rays_length
        sigma_delta = torch.nn.Softplus()(sigmas) * dist_samples
        alpha = 1 - torch.exp(-sigma_delta)
        T = torch.exp(-torch.cat([torch.zeros_like(sigma_delta[..., :1]), sigma_delta[..., :-1]], dim = 1).cumsum(dim = 1))
        prob = (T * alpha)[..., None]
        opacity = prob.sum(dim = 1)
        depth_samples = z_vals.unsqueeze(-1)
        depth = (depth_samples * prob).sum(dim = 1)
        weights = self.sigma2weights(deltas, sigmas)
        weights_sum = weights.sum(1) 
        rgbs_weights = weights.unsqueeze(-1)*rgbs
        rgb_final = torch.sum(rgbs_weights, 1)

        if self.white_back:
            rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)

        return rgb_final, sigmas, xyz, depth, opacity

    def sigma2weights(self, deltas, sigmas):
        noise = torch.randn(sigmas.shape, device=self.device)
        sigmas = sigmas + noise
        # [self.batch, 128]
        alphas = 1-torch.exp(-deltas*torch.nn.Softplus()(sigmas))
        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] 
        return weights

    def save_model(self, model, epoch):
        save_path = os.path.join(Path(self.weights_pth), Path("train"))        
        net  = "{}-EPOCH-{}-".format(self.data_name, epoch)
        save_dict = {'model_nerf': model.state_dict()}            
        nowtime = time.strftime("%Y-%m-%d-%H-%M-%S.ckpt", time.localtime())
        self.model_name = net + nowtime
        self.file_path = os.path.join(Path(save_path), Path(self.model_name))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if self.sys_param["distributed"]:
            if dist.get_rank() == 0:
                torch.save(save_dict, self.file_path)
        else:
            torch.save(save_dict, self.file_path)
        logging.info('\nSave model:{}'.format(self.model_name))
    
    def valid_train(self, epoch, val_data, epoch_type):
        if epoch_type in ['CAM_PARAM_EPOCH']:
            return 0
        torch.cuda.empty_cache()
        rank = get_rank()
        if rank == 0:
            rays_d, rays_o, gt_rgbs = val_data
            render_ckpt = torch.load(self.file_path, map_location = self.device)
            logging.info("Loading model：{}".format(self.model_name))
            val_nerf_coarse = CorseFine_NeRF(self.sys_param, type="coarse").to(self.device)
            val_nerf_fine = CorseFine_NeRF(self.sys_param, type="fine").to(self.device)
            nerf_ckpt_coarse = self.rewrite_nerf_ckpt(render_ckpt, coarse=True)
            nerf_ckpt_fine = self.rewrite_nerf_ckpt(render_ckpt, coarse=False)
            val_nerf_coarse.load_state_dict(nerf_ckpt_coarse)
            val_nerf_fine.load_state_dict(nerf_ckpt_fine)
            val_nerf_coarse.eval()
            val_nerf_fine.eval()
            rgb_cat = []
            depth_cat = []
            with torch.no_grad():
                logging.info("Rendering...")
                for ii in range(0, rays_d.shape[0], self.batch_test):
                    batch_rays_d = rays_d[ii:ii+self.batch_test]
                    batch_rays_o = rays_o[ii:ii+self.batch_test]
                    pred_rays_rgbs, pred_rays_depth, _ = self.render_rays_test(batch_rays_d, batch_rays_o, val_nerf_coarse, val_nerf_fine)
                    rgb_cat += [pred_rays_rgbs]
                    depth_cat += [pred_rays_depth] 
                rgb_cat = torch.cat(rgb_cat, 0)
                depth_cat = torch.cat(depth_cat, 0)
                logging.info("Saving image...")

            img = rgb_cat.view(self.render_h, self.render_w, 3).cpu()
            dep = depth_cat.view(self.render_h, self.render_w, 1).cpu()
            gt = gt_rgbs.view(self.render_h, self.render_w, 3).cpu()

            img = img.permute(2, 0, 1) # (3, H, W)
            dep = dep.permute(2, 0, 1)
            gt = gt.permute(2, 0, 1) # (3, H, W)

            img_path = os.path.join(Path(self.train_img_pth), Path(self.data_name), Path("epoch_"+ str(epoch) + ".png"))
            gt_path = os.path.join(Path(self.train_img_pth), Path(self.data_name), Path("epoch_"+ str(epoch) + "_gt.png"))
            dep_path = os.path.join(Path(self.train_img_pth), Path(self.data_name), Path("epoch_"+ str(epoch) + "_depth.png"))
            os.makedirs(os.path.dirname(img_path), exist_ok=True)

            transforms.ToPILImage()(img).convert("RGB").save(img_path)
            transforms.ToPILImage()(gt).convert("RGB").save(gt_path)
            transforms.ToPILImage()(dep).convert("L").save(dep_path)

            psnr_score = self.psnr_score(img, gt)
            lpips_score = self.lpips_score(img, gt)
            ssim_score = self.ssim_score(img, gt)

            logging.info("PSNR:{}".format(psnr_score))
            logging.info("LPIPS:{}".format(lpips_score))
            logging.info("SSIM:{}".format(ssim_score))

        if self.sys_param["distributed"]:
            dist.barrier()

        torch.cuda.empty_cache()

    def rewrite_nerf_ckpt(self, nerf_ckpt_dict, coarse=False):
        state_dict = nerf_ckpt_dict["model_nerf"]
        new_state_dict = state_dict.copy()
        if coarse:
            net_name = "nerf_coarse"
        else:
            net_name = "nerf_fine"
        for key in state_dict:
            new_key_name = ""
            if net_name in key.split("."):
                del(new_state_dict[key])
                idx_hash_nerf = key.split(".").index(net_name)
                new_key_name_list = key.split(".")[(idx_hash_nerf+1):]
                for idx, ele in enumerate(new_key_name_list):
                    new_key_name += ele
                    if idx != (len(new_key_name_list) - 1):
                        new_key_name += "."

                new_state_dict[new_key_name] = state_dict[key]
            else:
               del(new_state_dict[key])

        return new_state_dict
    
    def psnr_score(self, image_pred, image_gt, valid_mask=None, reduction='mean'):
        def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
            value = (image_pred-image_gt)**2
            if valid_mask is not None:
                value = value[valid_mask]
            if reduction == 'mean':
                return torch.mean(value)
            return value
        return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

    def lpips_score(self, image_pred, image_gt):
        lpips_loss = lpips.LPIPS(net="alex")
        lpips_value = lpips_loss(image_pred*2-1, image_gt*2-1).item()
        return lpips_value
    def ssim_score(self, image_pred, image_gt):
        image_pred = image_pred.unsqueeze(0)
        image_gt = image_gt.unsqueeze(0)
        ssim = pytorch_ssim.ssim(image_pred, image_gt).item()
        return ssim

    def query_sigma(self, xyz):
        ijk_coarse = ((xyz - self.xyz_min) / self.xyz_scope * self.grid_nerf).long().clamp(min=0, max=self.grid_nerf-1)
        sigmas = self.sigma_voxels[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]]
        
        return sigmas
    def update_sigma(self, xyz, sigma, beta):
        ijk_coarse = ((xyz - self.xyz_min) / self.xyz_scope * self.grid_nerf).long().clamp(min=0, max=self.grid_nerf-1)
        self.sigma_voxels[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]] \
                   = (1 - beta)*self.sigma_voxels[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]] + beta*sigma 
        