import logging
import time
import os

from pathlib import Path

class Log_config():
    def __init__(self, sys_param):
        self.save_mode = sys_param['log']
        self.save_pth = sys_param['log_pth']
        self.log_function_start()

    def log_function_start(self):
        if self.save_mode:
            results_log_pth = os.path.join(Path("results"), Path(self.save_pth))
            if not os.path.exists(results_log_pth):
                os.makedirs(results_log_pth)
            ticks = time.asctime(time.localtime(time.time()) )
            ticks = str(ticks).replace(' ', '-').replace(':','-')
            log_name = '{}.log'.format(os.path.join(self.save_pth, ticks))

            logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', 
                                datefmt='%m/%d/%Y %H:%M:%S', 
                                level=logging.INFO,
                                filemode='a',
                                filename=log_name)
        else:      
            logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', 
                                datefmt='%m/%d/%Y %H:%M:%S', 
                                level=logging.INFO)