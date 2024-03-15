import os
import torch
import logging

import torch.distributed as dist

class Distributed_config():
    def __init__(self, sys_param):
        self.init_distributed_mode(sys_param)
  
    def init_distributed_mode(self, sys_param):
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            sys_param['rank'] = int(os.environ["RANK"])
            sys_param['gpu'] = int(os.environ['LOCAL_RANK'])
            sys_param['world_size'] = int(os.environ['WORLD_SIZE'])

        else:
            logging.info('Muti-GPU Deactivate...')
            sys_param['distributed'] = False
            return

        start_device = sys_param["start_device"]
        sys_param['gpu'] = sys_param['gpu'] + start_device

        logging.info('Muti-GPU Activate: Current rank:{}, Current GPU:{}'.format(sys_param['rank'], sys_param['gpu']))
        sys_param['distributed'] = True
        torch.cuda.set_device(sys_param['gpu'])
        sys_param['dist_backend'] = 'nccl'
        torch.distributed.init_process_group(backend=sys_param['dist_backend'], 
                                             world_size=sys_param['world_size'], 
                                             rank=sys_param['rank'])

        torch.distributed.barrier(device_ids=[sys_param['gpu']])
        self.setup_for_distributed(sys_param['rank'] == 0)

    def setup_for_distributed(self, is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print
        logging_info = logging.info
        
        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)
        
        def info(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                logging_info(*args, **kwargs)            
     
        __builtin__.print = print
        logging.info = info

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True