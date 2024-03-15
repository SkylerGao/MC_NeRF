import os
import shutil 

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class Tensorboard_config():
    def __init__(self, sys_param):
        if sys_param['tb_available']:
            self.save_pth = os.path.join(Path("results"), Path(sys_param["tb_pth"]))
            if (sys_param["distributed"]) and (sys_param['rank'] == 0):
                if (sys_param['tb_del']):
                    if not os.path.exists(self.save_pth):
                        os.makedirs(self.save_pth)
                    else:
                        shutil.rmtree(self.save_pth)
                        os.makedirs(self.save_pth)
                else:
                    if not os.path.exists(self.save_pth):
                        os.makedirs(self.save_pth)
            self.tblogger = SummaryWriter(self.save_pth)
        else:
            self.tblogger = None
        
    def add_scalar(self, *args, **kwargs):
        self.tblogger.add_scalar(*args, **kwargs)