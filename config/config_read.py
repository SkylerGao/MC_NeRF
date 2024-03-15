import yaml
import os

from pathlib import Path
from utils import Log_config
from utils import Distributed_config

class Load_config():
    def __init__(self, args):
        # config file root
        self.root_config = args.config
        # terminal load
        self.system_info = self.load_terminal(args)
        # yaml load    
        self.load_yaml()
        # log config
        Log_config(self.system_info)   
        # muti GPU config
        Distributed_config(self.system_info)

    def load_yaml(self):
        path_yaml = os.path.join(Path(self.root_config), Path('config.yaml'))
        with open(path_yaml, 'r', encoding='utf-8') as f:
            cfg_info = yaml.load(f, Loader = yaml.FullLoader)

            ################# System #####################
            self.system_info['log_pth'] = cfg_info['system']['log_params']['logpath']
            self.system_info['root_weight'] = cfg_info['system']['weights_params']['root_weights']
            self.system_info['root_out'] = cfg_info['system']['out_params']['root_out']
            self.system_info['device_type'] = cfg_info['system']['device']['dev']
            self.system_info['seed'] = cfg_info['system']['data']['seed']
            self.system_info['tb_pth'] = cfg_info['system']['tensorboard_params']['tb_pth'] 
            self.system_info['tb_del'] = cfg_info['system']['tensorboard_params']['del_mode'] 

            ################# MC-NeRF #####################
            self.system_info["stage1_epoch"] = cfg_info['system']['epoch']['cam_param_stage']
            self.system_info["stage2_epoch"] = cfg_info['system']['epoch']['global_opt_stage']
            self.system_info["stage3_epoch"] = cfg_info['system']['epoch']['fine_tune_stage']
            self.system_info["stage1_lr"] = cfg_info['system']['train_params']['stage_1_lr']
            self.system_info["stage2_lr"] = cfg_info['system']['train_params']['stage_2_lr']
            self.system_info["stage3_lr"] = cfg_info['system']['train_params']['stage_3_lr']
            self.system_info["batch"] = cfg_info['system']['train_params']['batch']
            self.system_info["weight_d"] = cfg_info['system']['train_params']['weight_decay']
            self.system_info["warmup_epoch"] = cfg_info['system']['train_params']['warmup_epoch'] 
            self.system_info["barf_mask"]  = cfg_info['model']['barf']['barf_mask'] 
            self.system_info["barf_start"]  = cfg_info['model']['barf']['barf_start']
            self.system_info["barf_end"]  = cfg_info['model']['barf']['barf_end']
            self.system_info["apriltag_size"] = cfg_info['system']['apriltag']['tag_size']   
            self.system_info['res_h'] = cfg_info['system']['test_params']['resolution_h']
            self.system_info['res_w'] = cfg_info['system']['test_params']['resolution_w']
            self.system_info['demo_ckpt'] = cfg_info['system']['test_params']['nerf_model_name']
            self.system_info['demo_render_pth'] = os.path.join(Path(self.system_info['root_out']), 
                                                               Path(cfg_info['system']['out_params']['test_enerf_pth']))

            ################# NeRF #####################
            self.system_info["near"] = cfg_info['model']['nerf']['near']  
            self.system_info["far"]  = cfg_info['model']['nerf']['far']  
            self.system_info["samples"]  = cfg_info['model']['nerf']['samples']
            self.system_info["scale"]  = cfg_info['model']['nerf']['sample_scale']
            self.system_info["sample_weight_thresh"]  = cfg_info['model']['nerf']['weight_thresh']
            self.system_info["grid_nerf"] = cfg_info['model']['nerf']['grid_nerf']
            self.system_info["boader_min"] = cfg_info['model']['nerf']['global_boader_min']
            self.system_info["boader_max"] = cfg_info['model']['nerf']['global_boader_max']
            self.system_info["sigma_init"] = cfg_info['model']['nerf']['sigma_init']
            self.system_info["sigma_default"] = cfg_info['model']['nerf']['sigma_default']
            self.system_info["white_back"] = cfg_info['model']['nerf']['white_back']
            self.system_info["emb_freqs_xyz"] = cfg_info['model']['nerf']['emb_freqs_xyz']
            self.system_info['coarse_MLP_depth'] = cfg_info['model']['nerf']['coarse_MLP_depth']
            self.system_info['coarse_MLP_width'] = cfg_info['model']['nerf']['coarse_MLP_width']
            self.system_info['coarse_MLP_skip'] = cfg_info['model']['nerf']['coarse_MLP_skip']
            self.system_info['fine_MLP_depth'] = cfg_info['model']['nerf']['fine_MLP_depth']
            self.system_info['fine_MLP_width'] = cfg_info['model']['nerf']['fine_MLP_width']
            self.system_info['fine_MLP_skip'] = cfg_info['model']['nerf']['fine_MLP_skip']
            self.system_info["MLP_deg"] = cfg_info['model']['nerf']['MLP_deg']

    def load_terminal(self, args):
        system_info = {}
        for mode, flag in enumerate([args.train, args.demo]):
            if flag is True:
                system_info['mode'] = mode
                break
        system_info['log'] = args.log
        system_info['start_device'] = args.start_device    
        system_info['tb_available'] = args.tensorboard
        
        self.data_root = args.root_data
        self.data_name = args.data_name
        # root path of data
        system_info['data_name'] = self.data_name   
        # name of data
        system_info['data_root'] = os.path.join(Path(self.data_root), Path(self.data_name))

        return system_info 