import torch
import cv2
import logging
import os
import json
import math
import apriltag
import random

import numpy as np

from PIL import Image
from pathlib import Path
from torchvision import transforms as T
from torch.utils.data import DataLoader, DistributedSampler

class Data_set(torch.utils.data.Dataset):
    def __init__(self, system_param):
        self.system_param = system_param
        self.data_root = self.system_param["data_root"]
        self.data_name = self.system_param["data_name"]
        self.batch = self.system_param['batch']
        self.test_img_h = self.system_param['res_h']
        self.test_img_w = self.system_param['res_w']
        self.tag_size = self.system_param["apriltag_size"]
        self.transform = T.ToTensor()
        self.each_epoch, self.total_epoch, self.barf_start, self.barf_end = self.get_squence_info()
        # world points of calibration cube
        self.tag_world_pts = self.apriltag_gt_pts()  

        # load render datasets
        self.rgbs_train, self.pose_train, self.intr_train,  self.train_numb,\
        self.rgbs_test,  self.pose_test,  self.intr_test,  self.test_numb,\
        self.rgbs_val,   self.pose_val,   self.intr_val,  self.val_numb = self.load_blender_data_info()

        self.intr_inv_train =  self.inverse_intrinsic(self.intr_train)
        self.intr_inv_test  =  self.inverse_intrinsic(self.intr_test)
        self.intr_inv_val   =  self.inverse_intrinsic(self.intr_val)

        self.intr = [self.intr_train, self.intr_test, self.intr_val]
        self.intr_inv = [self.intr_inv_train, self.intr_inv_test, self.intr_inv_val]
        self.data_numb = [self.train_numb, self.test_numb, self.val_numb]

        self.train_idx = torch.arange(0, self.train_numb)
        self.test_idx = torch.arange(0, self.test_numb)
        self.val_idx = torch.arange(0, self.val_numb)

        # train mode
        if self.system_param["mode"] == 0:
            self.rgbs_expd, self.idx_expd = self.expand_data_length(self.rgbs_train,
                                                                    self.train_h,
                                                                    self.train_w, 
                                                                    self.train_idx,
                                                                    times=50)
            # load calibration info
            self.intr_wpts, self.intr_pts,\
            self.extr_wpts, self.extr_pts,\
            self.muti_tag_imgs = self.load_apriltag_json(self.data_root, self.expd_times)
        # test mode
        else:
            self.img_h = self.test_img_h
            self.img_w = self.test_img_w
            
        self.update_system_param = self.update_param(system_param) 

    def __len__(self):
        if self.system_param["mode"] == 0:
            return len(self.rgbs_expd)
        else:
            return self.test_numb
              
    def __getitem__(self, idx):
        if self.system_param["mode"] == 0:           
            return self.rgbs_expd[idx], self.idx_expd[idx], self.intr_wpts[idx], self.intr_pts[idx],\
                   self.extr_wpts[idx], self.extr_pts[idx]
        else:
            return self.rgbs_test[idx], self.test_idx[idx]
    
    # load MC datasets
    def load_blender_data_info(self):
        logging.info("Loading blender datasets...")
        logging.info("Current object:{}".format(self.data_name))
        # json file path
        self.train_json = os.path.join(Path(self.data_root), Path("transforms_train.json"))
        test_json  = os.path.join(Path(self.data_root), Path("transforms_test.json"))
        val_json   = os.path.join(Path(self.data_root), Path("transforms_val.json"))
        # json to camera and image info
        fov_train, img_pth_train, pose_train, train_numb = self.load_blender_json(self.train_json, self.data_root)
        fov_test,  img_pth_test,  pose_test, test_numb  = self.load_blender_json(test_json, self.data_root)
        fov_val,   img_pth_val,   pose_val, val_numb   = self.load_blender_json(val_json, self.data_root)
        # rgba images to rgb
        rgbs_train, self.train_h, self.train_w = self.preprocess_blender_images(img_pth_train)
        rgbs_test, test_h, test_w  = self.preprocess_blender_images(img_pth_test)
        rgbs_val, val_h, val_w   = self.preprocess_blender_images(img_pth_val)
        # camera fov to intrinsic mat
        intr_train = self.blender_fov_to_intrinsic(fov_train, self.train_h, self.train_w)
        intr_test  = self.blender_fov_to_intrinsic(fov_test, test_h, test_w)
        intr_val   = self.blender_fov_to_intrinsic(fov_val, val_h, val_w)

        return rgbs_train, pose_train, intr_train, train_numb,\
               rgbs_test, pose_test, intr_test, test_numb,\
               rgbs_val, pose_val, intr_val, val_numb

    def load_blender_json(self, json_path, root_path,  mode="extr"):
        with open(json_path,'r') as f:
            json_file = json.load(f)
        pose_list = []
        path_list = []
        fov_list = []

        for i,data in enumerate(json_file['frames']):
            img_path = os.path.join(Path(root_path), Path(data["file_path"] + ".png"))
            cam_angle_x = data["camera_angle_x"]
            if mode == "extr":
                pose = np.array(data['transform_matrix'])
                pose = torch.tensor(pose)
                pose = self.blender_pose_transform(pose) # [3, 4]
                pose_list += [pose]
            path_list += [img_path]
            fov_list += [cam_angle_x]
        if mode == "extr":
            pose_list = torch.stack(pose_list, 0)
        fov_tensor = torch.tensor(fov_list)
        data_numb = len(path_list)

        return fov_tensor, path_list, pose_list, data_numb

    # rgba to rgb
    def preprocess_blender_images(self, img_path):
        rgbs_list = []
        for pth in img_path:
            img = Image.open(pth)
            img = self.transform(img)
            img_h, img_w = img.shape[1], img.shape[2]
            img = img.reshape(4, -1).permute(1, 0) # (h*w, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:])
            rgbs_list += [img]
        rgbs_tensor = torch.stack(rgbs_list, 0)
        return rgbs_tensor, img_h, img_w

    def blender_fov_to_intrinsic(self, fov, img_h, img_w):
        intr_mat_list = []
        for f in fov:
            fx = (img_w/2)/(math.tan(f/2))
            fy = (img_h/2)/(math.tan(f/2))
            intr_mat = torch.tensor([[fx, 0, img_w/2],
                                     [0, fy, img_h/2],
                                     [0,  0,       1]])
            intr_mat_list += [intr_mat]
        intr_mat = torch.stack(intr_mat_list, 0)

        return intr_mat

    # load calibration datasets
    def load_apriltag_json(self, apriltag_root, times=1):
        logging.info("Loading calibration packages...") 
        calib_json = os.path.join(Path(apriltag_root), Path("transforms_calib.json"))
        _, path_calib, _, _ = self.load_blender_json(calib_json, apriltag_root, mode="intr")
        coord_json = os.path.join(Path(apriltag_root), Path("transforms_coord.json"))
        _, path_coord, _, _ = self.load_blender_json(coord_json, apriltag_root, mode="extr")
        tag_info_calib, tag_info_id_calib, muti_tag_id_calib, self.img_h, self.img_w = self.apriltag_detection(path_calib, check=True)
        tag_info_coord, tag_info_id_coord, muti_tag_id_coord, self.img_h, self.img_w = self.apriltag_detection(path_coord)
        # calibration data for intrinsic parameters 
        intr_wpts, intr_pts = self.get_cam_train_data(tag_info_calib, param="intr", times=times)
        # calibration data for extrinsic parameters 
        extr_wpts, extr_pts, extr_id_sq = self.get_cam_train_data(tag_info_coord, param="extr", times=times)
 
        return intr_wpts, intr_pts, extr_wpts, extr_pts, muti_tag_id_calib
        
    # apriltag detection, when check=True, muti-Apriltag detection is activated
    def apriltag_detection(self, apriltag_pth, check=False):
        all_tag_info = {}
        muti_tag_id = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
        all_id_info = {0:{'id':[],'pts':[]},\
                       1:{'id':[],'pts':[]},\
                       2:{'id':[],'pts':[]},\
                       3:{'id':[],'pts':[]},\
                       4:{'id':[],'pts':[]},\
                       5:{'id':[],'pts':[]}}
        detect_flag = 0
        detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))
        for img_id, pth in enumerate(apriltag_pth):
            cur_img = cv2.imread(pth)
            img_h, img_w = cur_img.shape[0], cur_img.shape[1]
            gray_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)  
            tags = detector.detect(gray_img)
            if len(tags) != 0:
                detect_flag += 1
                muti_tag_id[len(tags)] += [img_id]
            else:
                logging.info("Apriltags in image{} are not detected !!".format(img_id))
            tag_ids = []
            tag_pts = []
            for tag in tags:
                tag_id = tag.tag_id
                center_p = tag.center  # format:[x, y], also (w, h)
                corner_p = tag.corners # order:[lt, rt, rb， lb] 
                points_tag = np.concatenate([center_p.reshape([1, -1]), corner_p], 0)
                tag_pts += [points_tag]
                tag_ids += [tag_id]
                all_id_info[tag_id]['id'] += [img_id]
                all_id_info[tag_id]['pts'] += [points_tag] 
            if check and (len(tag_ids) < 2):
                logging.info("Muti-Apriltags are not detected in image{} !!".format(img_id))
            all_tag_info[img_id] = [tag_ids, tag_pts]
        
        if detect_flag == len(apriltag_pth):
            logging.info("All images include calibration points....")
        else:
            logging.info("Unvalid calibration images existing !!")
            exit()

        return all_tag_info, all_id_info, muti_tag_id, img_h, img_w

    # generate data for camera parameters training
    def get_cam_train_data(self, target_dict, param="intr", times=1):
        global_pts_list = []
        global_wpts_list = []
        global_id_list = []
        for i in range(self.train_numb*times):
            pts_list = []
            wpts_list = []
            id_list = []
            for img_id, info in target_dict.items():
                idx = random.randint(0, len(info[0])-1)
                tag_id = info[0][idx]
                tag_pt = info[1][idx]
                tag_wpts = self.tag_world_pts[tag_id]
                id_list += [tag_id]
                wpts_list += [tag_wpts]
                pts_list += [torch.from_numpy(tag_pt).to(torch.float32)]
            global_id_list += [torch.tensor(id_list)]
            global_wpts_list += [torch.stack(wpts_list, 0)]
            global_pts_list += [torch.stack(pts_list, 0)]
        global_tag_wpts = torch.stack(global_wpts_list, 0)
        global_tag_pts = torch.stack(global_pts_list, 0)
        tag_id_squence = torch.stack(global_id_list, 0)

        if param == "intr":
            return global_tag_wpts, global_tag_pts
        else:
            return global_tag_wpts, global_tag_pts, tag_id_squence
        
    # blender格式的位姿转换函数
    def blender_pose_transform(self, pose):
        pose_R = pose[:3, :3].to(torch.float32)
        pose_T = pose[:3, 3:].to(torch.float32)
        pose_flip_R = torch.diag(torch.tensor([1.0,-1.0,-1.0]))
        pose_flip_T = torch.zeros([3, 1])
        pose_R_new = pose_R @ pose_flip_R 
        pose_T_new = pose_R @ pose_flip_T + pose_T
        pose_R_new_inv = pose_R_new.T
        pose_T_new_inv = -pose_R_new_inv @ pose_T_new
        new_pose = torch.cat([pose_R_new_inv, pose_T_new_inv], -1)  
  
        return new_pose      
        
    def inverse_intrinsic(self, intr_mats):
        intr_inv_list = []
        for intr in intr_mats:
            intr_inv = intr.inverse()
            intr_inv_list += [intr_inv]
        intr_inv = torch.stack(intr_inv_list, 0)
        return intr_inv

    def update_param(self, system_param):
        system_param["intr_mat"] = self.intr
        system_param["intr_mat_inv"] = self.intr_inv
        system_param["data_numb"] = self.data_numb
        system_param["gt_pose"] = self.pose_train
        system_param["valid_pose"] = self.pose_val
        system_param["test_pose"] = self.pose_test
        system_param["valid_rgbs"] = self.rgbs_val
        system_param["data_img_h"] = self.img_h
        system_param["data_img_w"] = self.img_w
        system_param["train_json_file"] = self.train_json
        system_param["epoch_squence"] = self.each_epoch
        system_param["epoch_numb"] = self.total_epoch
        system_param["barf_start"] = self.barf_start
        system_param["barf_end"] = self.barf_end

        return system_param

    # expand datasets to have more training data in each epoch                                                                      
    def expand_data_length(self, rgbs, img_h, img_w, idx, times=None, squence=True):
        pixel_numb = img_h*img_w
        if times is None:
            self.expd_times = (pixel_numb // self.batch) + 1
        else:
            self.expd_times = times
        
        logging.info("Expanding datasets...")
        expd_rgbs = rgbs.repeat(self.expd_times, 1, 1)
        expd_idx = idx.repeat(self.expd_times)

        return expd_rgbs, expd_idx
    
    # generate world points for calibration cube
    def apriltag_gt_pts(self):
        cube_half = self.tag_size/2
        tag_half = self.tag_size*0.8/2
        world_tag_pts = {0:[[0.0,       -cube_half,       0.0],
                            [-tag_half, -cube_half,  tag_half],
                            [ tag_half, -cube_half,  tag_half],
                            [ tag_half, -cube_half, -tag_half],
                            [-tag_half, -cube_half, -tag_half]],
                         1:[[  cube_half,      0.0,       0.0],
                            [  cube_half,-tag_half,  tag_half],
                            [  cube_half, tag_half,  tag_half],
                            [  cube_half, tag_half, -tag_half],
                            [  cube_half,-tag_half, -tag_half]],
                         2:[[  0.0,      cube_half,       0.0],
                            [ tag_half,  cube_half,  tag_half],
                            [-tag_half,  cube_half,  tag_half],
                            [-tag_half,  cube_half, -tag_half],
                            [ tag_half,  cube_half, -tag_half]],
                         3:[[ -cube_half,      0.0,       0.0],
                            [ -cube_half, tag_half,  tag_half],
                            [ -cube_half,-tag_half,  tag_half],
                            [ -cube_half,-tag_half, -tag_half],
                            [ -cube_half, tag_half, -tag_half]],
                         4:[[  0.0,            0.0,  cube_half],
                            [-tag_half,   tag_half,  cube_half],
                            [ tag_half,   tag_half,  cube_half],
                            [ tag_half,  -tag_half,  cube_half],
                            [-tag_half,  -tag_half,  cube_half]],
                         5:[[  0.0,            0.0, -cube_half],
                            [-tag_half,  -tag_half, -cube_half],
                            [ tag_half,  -tag_half, -cube_half],
                            [ tag_half,   tag_half, -cube_half],
                            [-tag_half,   tag_half, -cube_half]]}
        world_tag_pts_tensor = {}
        for key in world_tag_pts:
            world_tag_pts_tensor[key] = torch.tensor(world_tag_pts[key])
        return world_tag_pts_tensor

    def get_squence_info(self):
        stage1_epoch = self.system_param['stage1_epoch']
        stage2_epoch = self.system_param['stage2_epoch']
        stage3_epoch = self.system_param['stage3_epoch']
        each_epoch = torch.tensor([stage1_epoch, stage2_epoch, stage3_epoch], dtype=torch.long)
        total_epoch = int(each_epoch.sum())
        barf_start = self.system_param["barf_start"]
        barf_end = self.system_param["barf_end"]
        global_barf_start = float(stage1_epoch)/float(total_epoch) + barf_start
        global_barf_end = float(stage1_epoch + stage2_epoch)/float(total_epoch)
        ratio = (global_barf_end - global_barf_start)*barf_end
        global_barf_end = global_barf_start + ratio
        
        return each_epoch, total_epoch, global_barf_start, global_barf_end


class Data_loader():
    def __init__(self, dataset, sys_param):
        self.dataset = dataset
        self.sys_param = sys_param
        if sys_param['distributed']:
            self.sampler = DistributedSampler(dataset, shuffle=True)
            self.sampler_no_shuffle = DistributedSampler(dataset, shuffle=False)
        else:
            self.sampler = torch.utils.data.RandomSampler(dataset)
            self.sampler_no_shuffle = torch.utils.data.SequentialSampler(dataset)
        
        self.dataloader = self.pkg_dataloader()

    def pkg_dataloader(self, batch=1):
        self.batch_sampler_train = torch.utils.data.BatchSampler(self.sampler, batch, drop_last=True)      
        self.batch_sampler_val   = torch.utils.data.BatchSampler(self.sampler_no_shuffle, batch, drop_last=False)
        
        loader_train = DataLoader(self.dataset,
                                batch_sampler=self.batch_sampler_train,
                                num_workers=12,
                                pin_memory=True)

        loader_val = DataLoader(self.dataset,
                                batch_sampler=self.batch_sampler_val,
                                num_workers=12,
                                pin_memory=True)        
        
        return {"Shuffle_loader": loader_train, "Squence_loader": loader_val}