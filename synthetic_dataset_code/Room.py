import bpy
import math
import json
import os
import random
import cv2

from pathlib import Path
import numpy as np

from mathutils import *

class Room_Dataset():
    def __init__(self, obj_name, seed):        
        self.object_name = obj_name
        self.fov_min = 40
        self.fov_max = 80
        self.numb_train_val = 88
        self.numb_test = 200
        self.res_x = 800
        self.res_y = 800
        self.root_path = "F:\\NERF_Dataset"
        self.room_x = 6
        self.room_y = 4
        self.room_z = 3
        self.theta_room = 15
        self.round_numb = 7
        self.radius = min(self.room_x, self.room_y, self.room_z)
        self.start_pos = np.array([0, -self.radius, 0])
        self.start_rot = np.array([self.angle2rad(90), 0, 0])
        random.seed(seed)
        self.cam_list = self.init_camera()
        self.train_fov, self.val_fov, self.test_fov = self.get_cam_fov_room()
        self.loc_train, self.loc_val, self.loc_test,\
        self.rot_train, self.rot_val, self.rot_test = self.get_cam_pose_room()
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
            cur_cam.name = "Room_{}".format(idx)
            cur_cam.data.type = 'PERSP'
            cur_cam.data.lens_unit = 'FOV'
            cam_list += [cur_cam]
            
        return cam_list

    def get_cam_fov_room(self):
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
            
    def get_cam_pose_room(self):
        loc_test = []
        rot_test = []
        numb_rot_loc  = 180 // self.theta_room
        numb_rot_rot  = 360 // self.theta_room  
        loc_train = self.rect_position(self.round_numb, numb_rot_loc)
        rot_train = self.rect_rotation(self.round_numb, numb_rot_rot, loc_train)
        loc_val, rot_val = self.surface_position() 
        theta = list(np.linspace(360, -360, self.numb_test))
        phi = list(np.linspace(90, 0, self.numb_test//2))
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

    def rect_position(self, numb_round, numb_rot):
        pos_m = []
        pos_t = []
        pos_b = []
        step_z = self.room_z/numb_round
        for i in range(1, numb_round-1):
            cur_z = step_z*i
            p1 =  np.array([self.room_x/2, 0, cur_z]) 
            p2 =  np.array([self.room_x/2, self.room_y/2, cur_z]) 
            p3 =  np.array([0, self.room_y/2, cur_z])
            p4 =  np.array([-self.room_x/2, self.room_y/2, cur_z])
            p5 =  np.array([-self.room_x/2, 0, cur_z])
            p6 =  np.array([-self.room_x/2, -self.room_y/2, cur_z])
            p7 =  np.array([0, -self.room_y/2, cur_z])
            p8 =  np.array([self.room_x/2, -self.room_y/2, cur_z])
            pos_m += [p1, p2, p3, p4, p5, p6, p7, p8]
        pos_temp1b = []
        pos_temp2b = []
        pos_temp1t = []
        pos_temp2t = []
        for i in range(numb_rot):
            rad_theta = self.angle2rad(self.theta_room*i)
            if (self.theta_room*i) == 90:
                p1b = np.array([0,  self.room_y/2, 0])
                p2b = np.array([0, -self.room_y/2, 0])
                p1t = np.array([0,  self.room_y/2, self.room_z])
                p2t = np.array([0, -self.room_y/2, self.room_z])
            elif (self.theta_room*i) == 0:
                p1b = np.array([self.room_x/2,  0, 0])
                p2b = np.array([-self.room_x/2, 0, 0])
                p1t = np.array([self.room_x/2,  0, self.room_z])
                p2t = np.array([-self.room_x/2, 0, self.room_z])
            else:
                x_abs =  self.room_y/(2*math.tan(rad_theta))
                symbol = x_abs/np.abs(x_abs)  
                if symbol > 0:   
                    y_abs =  math.tan(rad_theta)*self.room_x/2
                else:
                    y_abs =  -math.tan(rad_theta)*self.room_x/2
                if np.abs(x_abs) >= self.room_x/2 :
                    p1b = np.array([ self.room_x/2*symbol,  y_abs, 0])
                    p2b = np.array([-self.room_x/2*symbol, -y_abs, 0])
                    p1t = np.array([ self.room_x/2*symbol,  y_abs, self.room_z])
                    p2t = np.array([-self.room_x/2*symbol, -y_abs, self.room_z])
                else:
                    p1b = np.array([ x_abs,  self.room_y/2, 0])
                    p2b = np.array([-x_abs, -self.room_y/2, 0])
                    p1t = np.array([ x_abs,  self.room_y/2, self.room_z])
                    p2t = np.array([-x_abs, -self.room_y/2, self.room_z])
            pos_temp1b += [p1b] 
            pos_temp2b += [p2b]
            pos_temp1t += [p1t] 
            pos_temp2t += [p2t]
        pos_b = pos_temp1b + pos_temp2b
        pos_t = pos_temp1t + pos_temp2t
        position = pos_b + pos_m + pos_t

        return position
    
    def rect_rotation(self, numb_round, numb_rot, location):
        rotation_list = []
        rad_phi_list = []
        phi_list = []
        theta_list = []  
        rad_theta = self.angle2rad(self.theta_room)
        for loc in location:
            radius = math.sqrt(loc[0]**2 + loc[1]**2)
            rad_phi = math.atan(loc[2]/radius)
            phi_list += [round(self.rad2angle(rad_phi), 1)]
            rad_phi_list += [-rad_phi]
            
        for i in range(numb_round):
            if (i == 0):
                bound = True
                start_rot_theta = np.array([self.angle2rad(90), 0, self.angle2rad(90)])
                start_rot_phi = start_rot_theta + np.array([0, 0, 0])
                bound_phi = rad_phi_list[1:numb_rot]
            elif(i == numb_round-1):
                bound = True
                start_rot_theta = np.array([self.angle2rad(90), 0, self.angle2rad(90)])
                start_rot_phi = start_rot_theta + np.array([-math.atan(2*self.room_z/self.room_x), 0, 0])
                bound_phi = rad_phi_list[-numb_rot+1:]
            else:
                bound = False
                skip = False
                mid_phi = rad_phi_list[numb_rot+(i-1)*8 : numb_rot+(i-1)*8+8]
            for j in range(numb_rot):
                if bound:
                    rotation_list += [start_rot_phi]
                    if j == numb_rot-1:
                        theta_list += [j*self.theta_room]
                        continue
                    start_rot_theta = start_rot_theta + np.array([0, 0, rad_theta])
                    start_rot_phi = start_rot_theta + np.array([bound_phi[j], 0, 0])
                    theta_list += [j*self.theta_room]
                else:
                    if skip:
                        pass
                    else:
                        theta_t = math.atan(self.room_y/self.room_x)
                        rot1 = np.array([self.angle2rad(90), 0, self.angle2rad(90)]) + np.array([mid_phi[0], 0, 0])
                        rot2 = np.array([self.angle2rad(90), 0, self.angle2rad(90) + theta_t]) + np.array([mid_phi[1], 0, 0])
                        rot3 = np.array([self.angle2rad(90), 0, self.angle2rad(180)]) + np.array([mid_phi[2], 0, 0])
                        rot4 = np.array([self.angle2rad(90), 0, self.angle2rad(270) - theta_t]) + np.array([mid_phi[3], 0, 0])
                        rot5 = np.array([self.angle2rad(90), 0, self.angle2rad(270)]) + np.array([mid_phi[4], 0, 0])
                        rot6 = np.array([self.angle2rad(90), 0, self.angle2rad(270) + theta_t]) + np.array([mid_phi[5], 0, 0])
                        rot7 = np.array([self.angle2rad(90), 0, self.angle2rad(360)]) + np.array([mid_phi[6], 0, 0])
                        rot8 = np.array([self.angle2rad(90), 0, self.angle2rad(450) - theta_t]) + np.array([mid_phi[7], 0, 0])
                        rotation_list += [rot1, rot2, rot3, rot4, rot5, rot6, rot7, rot8]
                        theta_t = theta_t * 180 / math.pi
                        theta_t = int(theta_t)
                        theta_list += [0, theta_t, 90, 180-theta_t, 180, 180+theta_t, 270, 360-theta_t]
                        skip = True
  
        return  rotation_list #, theta_list, phi_list

    def surface_position(self):
        loc_list = []
        rot_list = []
        cur_loc_list = []
        room_rx = self.room_x/2
        room_ry = self.room_y/2
        for i in range(self.numb_train_val):
            loc_ax = random.choice([0, 1, 2, 3, 4])
            if loc_ax == 0:
                cur_loc = [random.uniform(-room_rx, room_rx), random.uniform(-room_ry, room_ry), self.room_y] 
                while cur_loc in cur_loc_list:
                    cur_loc = [random.uniform(-room_rx, room_rx), random.uniform(-room_ry, room_ry), self.room_y] 
            if loc_ax == 1:
                cur_loc = [random.uniform(-room_rx, room_rx), -room_ry, random.uniform(0, self.room_z)]
                while cur_loc in cur_loc_list:
                    cur_loc = [random.uniform(-room_rx, room_rx), -room_ry, random.uniform(0, self.room_z)]
            if loc_ax == 2:
                cur_loc = [room_rx, random.uniform(-room_ry, room_ry), random.uniform(0, self.room_z)]
                while cur_loc in cur_loc_list:
                    cur_loc = [room_rx, random.uniform(-room_ry, room_ry), random.uniform(0, self.room_z)]
            if loc_ax == 3:
                cur_loc = [random.uniform(-room_rx, room_rx),  room_ry, random.uniform(0, self.room_z)]
                while cur_loc in cur_loc_list:
                    cur_loc = [random.uniform(-room_rx, room_rx),  room_ry, random.uniform(0, self.room_z)]
            if loc_ax == 4:
                cur_loc = [-room_rx, random.uniform(-room_ry, room_ry), random.uniform(0, self.room_z)]
                while cur_loc in cur_loc_list:
                    cur_loc = [-room_rx, random.uniform(-room_ry, room_ry), random.uniform(0, self.room_z)]

            cur_loc_list += [cur_loc]
            cur_theta, cur_phi = self.get_rot_from_loc(cur_loc)
            cur_rot  = self.start_rot + np.array([-cur_phi, 0, 0])
            cur_rot  = cur_rot + np.array([0, 0, cur_theta])
            loc_list += [cur_loc]
            rot_list += [cur_rot]

        return loc_list, rot_list

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
        return rot_theta, rot_phi
        
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

    def render_calibration_images(self, img_id=None):
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
        
        if img_id == None:
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
        else:
            fov = self.train_fov[img_id]
            cam = self.cam_list[:self.numb_train_val][img_id]
            bpy.context.scene.camera = cam
            loc, rot = self.loc_train[img_id], self.rot_train[img_id]
            cam.rotation_mode = 'XYZ'   
            cam.location = loc
            cam.rotation_euler = rot
            cam.data.angle = self.angle2rad(fov)
            cur_pose = cam.matrix_world
            save_path = os.path.join(Path(calib_path), Path("r_{}".format(img_id)))
            bpy.context.scene.render.filepath = save_path
            bpy.ops.render.render(write_still=True)
            while not self.apriltag_more_than_two(detector, save_path):
                object_cube.rotation_euler[0] = random.uniform(0, 2*math.pi)
                object_cube.rotation_euler[1] = random.uniform(0, 2*math.pi)
                object_cube.rotation_euler[2] = random.uniform(0, 2*math.pi)
                bpy.ops.render.render(write_still=True)
            frame_data = {'file_path': "./calib/r_{}".format(img_id),
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
    cur_data = "Snowtruck"
    dataset = Room_Dataset(obj_name = "Room_{}".format(cur_data), seed=seed_dict[cur_data])
    dataset.render_images()
    dataset.render_calibration_images()
    dataset.cam_clear()
