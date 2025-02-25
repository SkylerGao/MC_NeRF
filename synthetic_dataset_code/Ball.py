import bpy
import math
import json
import os
import random
import cv2

from pathlib import Path
import numpy as np

from mathutils import *

class Ball_Dataset():
    def __init__(self, obj_name, seed):
        self.object_name = obj_name
        self.fov_min = 40
        self.fov_max = 80
        self.numb_train_val = 110
        self.numb_test = 200
        self.res_x = 800
        self.res_y = 800
        self.root_path = "F:\\NERF_Dataset"
        self.radius = 3
        self.start_pos = np.array([0, -self.radius, 0])
        self.start_rot = np.array([self.angle2rad(90), 0, 0])
        random.seed(seed)
        self.cam_list = self.init_camera()
        self.train_fov, self.val_fov, self.test_fov = self.get_cam_fov_ball()
        self.loc_train, self.loc_val, self.loc_test,\
        self.rot_train, self.rot_val, self.rot_test = self.get_cam_pose_ball()
        self.render_set()


    def render_images(self):
        collect_objects = bpy.data.collections
        for collects in collect_objects:
            collects.hide_render = True
        collect_objects["Object"].hide_render = False
        collect_objects[self.object_name.split("_")[-1]].hide_render = False
        self.render_process()

    def render_process(self):
        train_path = os.path.join(Path(self.root_path), Path(self.object_name), Path("train"))
        val_path = os.path.join(Path(self.root_path), Path(self.object_name), Path("val"))
        test_path = os.path.join(Path(self.root_path), Path(self.object_name), Path("test"))
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(val_path):
            os.makedirs(val_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        for idx, cam_info in enumerate(zip(self.train_fov, self.cam_list[:self.numb_train_val])):
            fov, cam = cam_info
            bpy.context.scene.camera = cam
            loc, rot = self.loc_train[idx], self.rot_train[idx]
            cam.rotation_mode = 'XYZ'   
            cam.location = loc
            cam.rotation_euler = rot
            cam.data.angle = self.angle2rad(fov)
            cur_pose = cam.matrix_world
            save_path = os.path.join(Path(train_path), Path("r_{}".format(idx)))
            bpy.context.scene.render.filepath = save_path
            bpy.ops.render.render(write_still=True)
            frame_data = {'file_path': "./train/r_{}".format(idx),
                          "camera_angle_x": self.angle2rad(fov),
                          'transform_matrix': self.listify_matrix(cur_pose)}
            self.out_data_train['frames'].append(frame_data)
        with open(self.json_train_path, 'w') as out_file:
            json.dump(self.out_data_train, out_file, indent=4)
        for idx, cam_info in enumerate(zip(self.val_fov, self.cam_list[:self.numb_train_val])):
            fov, cam = cam_info
            bpy.context.scene.camera = cam
            loc, rot = self.loc_val[idx], self.rot_val[idx]
            cam.rotation_mode = 'XYZ'   
            cam.location = loc
            cam.rotation_euler = rot
            cam.data.angle = self.angle2rad(fov)
            cur_pose = cam.matrix_world
            save_path = os.path.join(Path(val_path), Path("r_{}".format(idx)))
            bpy.context.scene.render.filepath = save_path
            bpy.ops.render.render(write_still=True)
            frame_data = {'file_path': "./val/r_{}".format(idx),
                          "camera_angle_x": self.angle2rad(fov),
                          'transform_matrix': self.listify_matrix(cur_pose)}
            self.out_data_val['frames'].append(frame_data)
        with open(self.json_val_path, 'w') as out_file:
            json.dump(self.out_data_val, out_file, indent=4)
        for idx, cam_info in enumerate(zip(self.test_fov, self.cam_list)):
            fov, cam = cam_info
            bpy.context.scene.camera = cam
            loc, rot = self.loc_test[idx], self.rot_test[idx]
            cam.rotation_mode = 'XYZ'   
            cam.location = loc
            cam.rotation_euler = rot
            cam.data.angle = self.angle2rad(fov)
            cur_pose = cam.matrix_world
            save_path = os.path.join(Path(test_path), Path("r_{}".format(idx)))
            bpy.context.scene.render.filepath = save_path
            bpy.ops.render.render(write_still=True)
            frame_data = {'file_path': "./test/r_{}".format(idx),
                          "camera_angle_x": self.angle2rad(fov),
                          'transform_matrix': self.listify_matrix(cur_pose)}
            self.out_data_test['frames'].append(frame_data)
        with open(self.json_test_path, 'w') as out_file:
            json.dump(self.out_data_test, out_file, indent=4)

    def angle2rad(self, angle):
        return angle*math.pi/180

    def listify_matrix(self, matrix):
        matrix_list = []
        for row in matrix:
            matrix_list.append(list(row))
        return matrix_list

    def Rot_Z(self, angle_theta):
        rad_theta = self.angle2rad(angle_theta)
        rot_M_z = np.array([[np.cos(rad_theta), np.sin(rad_theta), 0],
                            [-np.sin(rad_theta), np.cos(rad_theta), 0],
                            [0, 0, 1]])    
        return rot_M_z

    def Rot_X(self, angle_phi):
        rad_phi = -self.angle2rad(angle_phi)
        rot_M_x = np.array([[1, 0, 0],
                        [0, np.cos(rad_phi), np.sin(rad_phi)],
                        [0, -np.sin(rad_phi), np.cos(rad_phi)]])
        return rot_M_x

    def init_camera(self):
        cam_list = []
        for idx in range(max(self.numb_train_val, self.numb_test)):
            bpy.ops.object.camera_add(enter_editmode=False, 
                                      align='VIEW', 
                                      location=(0, 0, 0), 
                                      rotation=(0, 0, 0),
                                      scale=(1, 1, 1))     
            cur_cam = bpy.context.selected_objects[0]
            cur_cam.name = "Ball_{}".format(idx)
            cur_cam.data.type = 'PERSP'
            cur_cam.data.lens_unit = 'FOV'
            cam_list += [cur_cam]
            
        return cam_list

    def get_cam_fov_ball(self):
        fov_angle_train = []
        fov_angle_val = []
        for i in range(self.numb_train_val):
            cur_fov_train = random.randint(self.fov_min, self.fov_max)
            while cur_fov_train in [fov_angle_train]:
                cur_fov_train = random.randint(self.fov_min, self.fov_max)
            fov_angle_train += [cur_fov_train]
        fov_angle_val = fov_angle_train
        fov_angle_test = list(np.linspace(self.fov_max, self.fov_min, self.numb_test//2))
        fov_angle_test_inv = fov_angle_test.copy()
        fov_angle_test_inv.sort()
        fov_angle_test = fov_angle_test + fov_angle_test_inv
        assert len(fov_angle_test) == self.numb_test, "Length Error for test fov !!!"
        
        return fov_angle_train, fov_angle_val, fov_angle_test
            
    def get_cam_pose_ball(self):
        loc_train, loc_val, loc_test = [], [], []
        rot_train, rot_val, rot_test = [], [], []
        theta_train = []
        phi_train = []
        theta_range = list(np.linspace(0, 360, 12, endpoint=False))
        phi_range = list(np.linspace(-80, 80, 9))
        phi_end = [-90, 90]
        for phi in phi_range:
            roll_mat = self.Rot_X(phi)
            for theta in theta_range:
                theta_train += [theta]
                phi_train += [phi]
                pitch_mat = self.Rot_Z(theta)
                next_pose = np.matmul(self.start_pos, roll_mat)
                next_pose = np.matmul(next_pose, pitch_mat)
                next_rot  = self.start_rot + np.array([-self.angle2rad(phi), 0, 0])
                next_rot  = next_rot + np.array([0, 0, self.angle2rad(theta)])
                loc_train += [next_pose]
                rot_train += [next_rot]
        for phi in phi_end:
            roll_mat = self.Rot_X(phi)
            theta_train += [0] # theta没转
            phi_train += [phi]            
            next_pose = np.matmul(self.start_pos, roll_mat)
            next_rot  = self.start_rot + np.array([-self.angle2rad(phi), 0, 0])
            loc_train += [next_pose]
            rot_train += [next_rot]           

        for i in range(self.numb_train_val):
            theta = random.randint(0, 360)
            phi = random.randint(0, 90)
            while theta in theta_train:
                theta = random.randint(0, 360)
            while phi in phi_train:
                phi = random.randint(0, 90)
            pitch_mat = self.Rot_Z(theta)
            roll_mat = self.Rot_X(phi)
            next_pose = np.matmul(self.start_pos, roll_mat)
            next_pose = np.matmul(next_pose, pitch_mat)
            next_rot  = self.start_rot + np.array([-self.angle2rad(phi), 0, 0])
            next_rot  = next_rot + np.array([0, 0, self.angle2rad(theta)])       
            loc_val += [next_pose]
            rot_val += [next_rot]
        theta = list(np.linspace(360, -360, self.numb_test))
        phi = list(np.linspace(90, -90, self.numb_test//2))
        phi_inv = phi.copy()
        phi_inv.sort()
        phi = phi + phi_inv

        assert len(theta) == self.numb_test, "Length Error for test pose !!!"
        for i in range(self.numb_test):
            pitch_mat = self.Rot_Z(theta[i])
            roll_mat = self.Rot_X(phi[i])
            next_pose = np.matmul(self.start_pos, roll_mat)
            next_pose = np.matmul(next_pose, pitch_mat)
            next_rot  = self.start_rot + np.array([-self.angle2rad(phi[i]), 0, 0])
            next_rot  = next_rot + np.array([0, 0, self.angle2rad(theta[i])])       
            loc_test += [next_pose]
            rot_test += [next_rot]

        return  loc_train, loc_val, loc_test, rot_train, rot_val, rot_test

    def render_set(self):
        self.json_train_path = os.path.join(Path(self.root_path), Path(self.object_name), Path("transforms_train.json"))
        self.json_val_path   = os.path.join(Path(self.root_path), Path(self.object_name), Path("transforms_val.json"))
        self.json_test_path  = os.path.join(Path(self.root_path), Path(self.object_name), Path("transforms_test.json"))
        self.json_coord_path  = os.path.join(Path(self.root_path), Path(self.object_name), Path("transforms_coord.json"))
        self.json_calib_path  = os.path.join(Path(self.root_path), Path(self.object_name), Path("transforms_calib.json"))
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.resolution_x = self.res_x
        bpy.context.scene.render.resolution_y = self.res_y
        self.out_data_train = {"frames":[]}
        self.out_data_val = {"frames":[]}
        self.out_data_test = {"frames":[]}
        self.out_data_coord = {"frames":[]}
        self.out_data_calib = {"frames":[]}

    def cam_clear(self):
        for cam in self.cam_list:
            bpy.data.objects.remove(cam)

    def apriltag_more_than_two(self, detector, save_path):
        save_path = save_path + ".png"
        cur_img = cv2.imread(save_path) 
        gray_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) 
        _, ids, _ = detector.detectMarkers(gray_img)
        if (ids is not None) and (len(ids) > 2):
            return True
        else:
            return False

    def render_calibration_images(self):
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        arucoParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
        coord_path = os.path.join(Path(self.root_path), Path(self.object_name), Path("coord"))
        calib_path = os.path.join(Path(self.root_path), Path(self.object_name), Path("calib"))
        if not os.path.exists(coord_path):
            os.makedirs(coord_path)
        if not os.path.exists(calib_path):
            os.makedirs(calib_path)
        collect_objects = bpy.data.collections
        for collects in collect_objects:
            collects.hide_render = True
        collect_objects["Calibration Object"].hide_render = False
        object_cube = bpy.data.objects["Cube"]
        object_cube.rotation_euler = [0, 0, 0]
        for idx, cam_info in enumerate(zip(self.train_fov, self.cam_list[:self.numb_train_val])):
            fov, cam = cam_info
            bpy.context.scene.camera = cam
            loc, rot = self.loc_train[idx], self.rot_train[idx]
            cam.rotation_mode = 'XYZ'   
            cam.location = loc
            cam.rotation_euler = rot
            cam.data.angle = self.angle2rad(fov)
            cur_pose = cam.matrix_world
            save_path = os.path.join(Path(coord_path), Path("r_{}".format(idx)))
            bpy.context.scene.render.filepath = save_path
            bpy.ops.render.render(write_still=True)
            frame_data = {'file_path': "./coord/r_{}".format(idx),
                          "camera_angle_x": self.angle2rad(fov),
                          'transform_matrix': self.listify_matrix(cur_pose)}
            self.out_data_coord['frames'].append(frame_data)
        with open(self.json_coord_path, 'w') as out_file:
            json.dump(self.out_data_coord, out_file, indent=4)

        for idx, cam_info in enumerate(zip(self.train_fov, self.cam_list[:self.numb_train_val])):
            fov, cam = cam_info
            bpy.context.scene.camera = cam
            loc, rot = self.loc_train[idx], self.rot_train[idx]
            cam.rotation_mode = 'XYZ'   
            cam.location = loc
            cam.rotation_euler = rot
            cam.data.angle = self.angle2rad(fov)
            save_path = os.path.join(Path(calib_path), Path("r_{}".format(idx)))
            bpy.context.scene.render.filepath = save_path
            bpy.ops.render.render(write_still=True)
            while not self.apriltag_more_than_two(detector, save_path):
                object_cube.rotation_euler[0] = random.uniform(0, 2*math.pi)
                object_cube.rotation_euler[1] = random.uniform(0, 2*math.pi)
                object_cube.rotation_euler[2] = random.uniform(0, 2*math.pi)
                bpy.ops.render.render(write_still=True)

            frame_data = {'file_path': "./calib/r_{}".format(idx),
                          "camera_angle_x": self.angle2rad(fov)}
            self.out_data_calib['frames'].append(frame_data)
        with open(self.json_calib_path, 'w') as out_file:
            json.dump(self.out_data_calib, out_file, indent=4)
        collect_objects["Calibration Object"].hide_render = True
                
if __name__ == "__main__":
    seed_dict = {"Lego":0,
                 "Gate":1,
                 "Materials":2,
                 "Ficus":3,
                 "Computer":4,
                 "Snowtruck":5,
                 "Statue":6,
                 "Train":7}
    cur_data = "Gate"
    dataset = Ball_Dataset(obj_name = "Ball_{}".format(cur_data), seed=seed_dict[cur_data])
    dataset.render_images()
    dataset.render_calibration_images()
    dataset.cam_clear()