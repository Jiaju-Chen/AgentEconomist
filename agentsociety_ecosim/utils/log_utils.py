# agentsociety_ecosim/utils/log_utils.py
import logging
import os
from datetime import datetime
import sys

log_file_path = None 
def setup_global_logger(name="ecosim", log_dir="logs", level=logging.INFO):
    global log_file_path
    os.makedirs(log_dir, exist_ok=True)

    if log_file_path is None:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f"{name}_{time_str}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  

    if not logger.handlers:
        # 文件输出
        fh = logging.FileHandler(log_file_path)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        logger.addHandler(fh)

        # 控制台输出
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        logger.addHandler(sh)

    return logger

def get_logger():
    return logging.getLogger("ecosim")
