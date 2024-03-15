import torch
import torch.nn as nn

class MC_NeRF_Loss(nn.Module):
    def __init__(self, sys_param, tblogger=None):
        super(MC_NeRF_Loss, self).__init__()
        self.sys_param = sys_param
        self.loss_l2 = nn.MSELoss(reduction='mean')
        # tensorboard
        self.tblogger = tblogger
        self.global_step = 0
        self.img_h = self.sys_param["data_img_h"]
        self.img_w = self.sys_param["data_img_w"]

    def forward(self, loss_dict, epoch_type):
        final_loss = 0.0
        self.global_step += 1
        if "intr" in loss_dict:
            loss_intr = self.get_reproject_loss(loss_dict["intr"])
            if epoch_type == "CAM_PARAM_EPOCH":
                final_loss += loss_intr
            else:
                final_loss += loss_intr/(loss_intr.detach() + 1e-8)
        if "extr" in loss_dict:
            loss_extr = self.get_reproject_loss(loss_dict["extr"])
            final_loss += loss_extr
        if "rgb" in loss_dict:
            loss_rgb = self.get_rgb_loss(loss_dict["rgb"])
            final_loss += loss_rgb

        return final_loss
    
    def get_rgb_loss(self, rgbs_list):
        rgbs_c = rgbs_list[0]
        rgbs_f = rgbs_list[1]
        rgbs_gt = rgbs_list[2]
        if rgbs_f is None:
            loss_rgb_f = 0
        else:
            loss_rgb_f = self.loss_l2(rgbs_f, rgbs_gt)
        loss_rgb_c = self.loss_l2(rgbs_c, rgbs_gt)
        loss = loss_rgb_c + loss_rgb_f
        return loss

    def get_reproject_loss(self, rpro_list):
        # pts format [x, y]
        pd_pts = rpro_list[0]
        gt_pts = rpro_list[1]
        
        pd_pts_nx = pd_pts[..., 0]
        pd_pts_ny = pd_pts[..., 1]
        gt_pts_nx = gt_pts[..., 0]
        gt_pts_ny = gt_pts[..., 1]
 
        proj_loss_x = self.loss_l2(pd_pts_nx/self.img_w, gt_pts_nx/self.img_w)
        proj_loss_y = self.loss_l2(pd_pts_ny/self.img_h, gt_pts_ny/self.img_h)

        return proj_loss_x + proj_loss_y

