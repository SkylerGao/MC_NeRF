import bpy
import math
import json
import os
import random
import cv2

from pathlib import Path
import numpy as np

from mathutils import *

class Array_Dataset():
    def __init__(self, obj_name, seed):
        self.object_name = obj_name
        self.fov_min = 40
        self.fov_max = 80
        self.res_x = 800
        self.res_y = 800
        self.root_path = "F:\\NERF_Dataset"
        self.array_x = 3
        self.array_y = 3
        self.radius_array = 4
        self.theta_array = 45
        self.numb_x = 10
        self.numb_y = 10
        self.numb_train_val = self.numb_x*self.numb_y
        self.numb_test = 200
        random.seed(seed)
        self.cam_list = self.init_camera()
        self.train_fov, self.val_fov, self.test_fov = self.get_cam_fov_array()
        self.loc_train, self.loc_val, self.loc_test,\
        self.rot_train, self.rot_val, self.rot_test = self.get_cam_pose_array()
        self.render_set()

    def render_images(self):
        collect_objects = bpy.data.collections
        for collects in collect_objects:
            collects.hide_render = True
        collect_objects["Object"].hide_render = False
        collect_objects[self.object_name.split("_")[-1]].hide_render = False
        self.render_process()
        self.cam_clear()

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
    
    def rad2angle(self, rad):
        return rad*180/math.pi 
    
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
            cur_cam.name = "Array_{}".format(idx)
            cur_cam.data.type = 'PERSP'
            cur_cam.data.lens_unit = 'FOV'
            cam_list += [cur_cam]
            
        return cam_list

    def get_cam_fov_array(self):
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
            
    def get_cam_pose_array(self):
        loc_train, rot_train = self.get_array_pose_train()
        loc_val, rot_val = self.get_array_pose_val()
        loc_test, rot_test = self.get_array_pose_test()

        return  loc_train, loc_val, loc_test, rot_train, rot_val, rot_test

    def get_array_pose_train(self):
        rot_M_x_90 = self.Rot_X(90)
        rot_M_z = self.Rot_Z(self.theta_array) 
        x_range = np.linspace(-self.array_x/2 , self.array_x/2, self.numb_x)
        y_range = np.linspace(-self.array_y/2 , self.array_y/2, self.numb_y)
        loc_x, loc_y = np.meshgrid(x_range, y_range)
        loc_z = -np.ones_like(loc_x)*self.radius_array
        cord_array = np.stack([loc_x, loc_y, loc_z], -1)
        cord_array = cord_array.reshape(-1, 3)
        cord_array = np.matmul(cord_array, rot_M_x_90)
        cord_array = np.matmul(cord_array, rot_M_z)

        loc_train = [cord for cord in cord_array]
        rot_train = [self.get_rot_from_loc(loc) for loc in loc_train]

        return loc_train, rot_train

    def get_array_pose_val(self):
        cord_list = []
        rot_M_x_90 = self.Rot_X(90)
        rot_M_z = self.Rot_Z(self.theta_array) 
        for i in range(self.numb_train_val):
            cur_x = random.uniform(-self.array_x/2 , self.array_x/2)
            cur_y = random.uniform(-self.array_y/2 , self.array_y/2)
            cur_z = -self.radius_array
            cord_list += [np.array([cur_x, cur_y, cur_z])]
        cord_array = np.stack(cord_list, 0)
        cord_array = np.matmul(cord_array, rot_M_x_90)
        cord_array = np.matmul(cord_array, rot_M_z)
        loc_val = [cord for cord in cord_array]
        rot_val = [self.get_rot_from_loc(loc) for loc in loc_val]

        return loc_val, rot_val

    def get_array_pose_test(self):
        cord_list = []
        rot_M_x_90 = self.Rot_X(90)
        rot_M_z = self.Rot_Z(self.theta_array)
        min_r = min(self.array_x, self.array_y)
        r_face = np.abs(np.linspace(min_r, -min_r, self.numb_test))
        rot_face = np.linspace(-360, 360, self.numb_test)
        for i in range(self.numb_test):
            start_pos = np.array([0.0, r_face[i], -self.radius_array])
            roll_mat = self.Rot_Z(rot_face[i])
            next_pose = np.matmul(start_pos, roll_mat)
            cord_list += [next_pose]
            
        cord_array = np.stack(cord_list, 0)
        cord_array = np.matmul(cord_array, rot_M_x_90)
        cord_array = np.matmul(cord_array, rot_M_z)
        
        loc_test = [cord for cord in cord_array]
        rot_test = [self.get_rot_from_loc(loc) for loc in loc_test]

        return loc_test, rot_test        

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

    def get_rot_from_loc(self, loc):
        loc_np = np.array(loc)
        loc_r = np.linalg.norm(loc_np[:2], ord=2)
        rot_phi = np.arctan(loc_np[2]/loc_r)
        loc_vect_xy = loc_np[:2]/loc_r
        std_vect = np.array([0, -1])
        cos_theta = np.dot(loc_vect_xy, std_vect) / (np.linalg.norm(loc_vect_xy)*np.linalg.norm(std_vect))
        sin_theta = np.cross(loc_vect_xy, std_vect) / (np.linalg.norm(loc_vect_xy)*np.linalg.norm(std_vect))
        if sin_theta > 0:
            rot_theta = 2*np.pi - np.arccos(cos_theta)
        else:
            rot_theta = np.arccos(cos_theta)
        rot = np.array([self.angle2rad(90), 0, 0]) + np.array([-rot_phi, 0, 0])
        rot = rot + np.array([0, 0, rot_theta])

        return rot
    
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
    cur_data = "Materials"
    dataset = Array_Dataset(obj_name = "Array_{}".format(cur_data), seed=seed_dict[cur_data])
    dataset.render_images()
    dataset.render_calibration_images()
    dataset.cam_clear()