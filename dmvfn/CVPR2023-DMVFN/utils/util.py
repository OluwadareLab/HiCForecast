import logging
import numpy as np
import os
import datetime

def get_timestamp():
    desired_timezone = datetime.timezone(datetime.timedelta(hours=-6))
    current_time = datetime.datetime.now(desired_timezone)
    formatted_time = current_time.strftime('%Y%m%d-%H%M%S')
    return formatted_time

def get_formatted_timestamp():
    desired_timezone = datetime.timezone(datetime.timedelta(hours=-6))
    current_time = datetime.datetime.now(desired_timezone)
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_time

def setup_logger(logger_name, save_dir, phase, level=logging.INFO, screen=False, to_file=False):
    lg = logging.getLogger(logger_name)
    #formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
    #                              datefmt='%y-%m-%d %H:%M:%S')

    formatter = logging.Formatter('%(levelname)s: %(message)s')
    lg.setLevel(level)
    if to_file:
        log_file = os.path.join(save_dir, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
        
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def rgb2ycbcr(img_np):
    h, w, _ = img_np.shape
    y_map = np.zeros((h, w)).astype(np.float32)
    Y = 0.257*img_np[:,:, 2]+0.504*img_np[:,:, 1]+0.098*img_np[:,:, 0]+16

    return Y
