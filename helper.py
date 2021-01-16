import os
import time
import shutil
import re

from collections import defaultdict

__all__ = ["parse_image_path_and_copy", "clean_folder", "convert_to_timestep"]


def parse_image_path_and_copy(image_paths, path_to_static_images):
    dct_camid_frameid = defaultdict(list)
    pattern = re.compile(r"[\d]{4,4}_c([\d]{1,1})_f([\d]{7,7}).jpg")
    
    for image_path in image_paths:
        fname = os.path.basename(image_path)
        shutil.copy(image_path, os.path.join(path_to_static_images, fname))
        match = pattern.match(fname)     # camid, frame
        if match:
            (camid,frameid) = match.groups()
            dct_camid_frameid[camid].append((camid, convert_to_timestep(frameid), fname))

    for key in dct_camid_frameid:
        dct_camid_frameid[key].sort()

    return [dct_camid_frameid[camid] for camid in dct_camid_frameid]

def convert_to_timestep(frameid):
    """ Convert frameid to timestep. Assumming the fps is 25 and the start time is 3 hour before now.
    :param:
        :frameid: string object; format as '0000123'
    :return: timestep; format as 'HH:MM:SS'
    """
    base_time_step = time.time() - 3600*3          # 以3小时前计算, dukemtmc中帧数最大227476, 按每秒25帧, 最多经过2.53小时
    frame = int(frameid)
    timestep = base_time_step + 0.04*frame
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestep))


def clean_folder(path_dir):
    """ remove all files in target folder
    :param:
        :path_dir: target folder
    """
    files = os.listdir(path_dir)
    for file in files:
        fpath = os.path.join(path_dir, file)
        if os.path.isdir(fpath):
            clean_folder(fpath)
            os.rmdir(fpath)
        else:
            os.remove(fpath)


# def return_image_stream(img_local_path):
#     """工具函数, 获取本地图片流
#     :param
#         :img_local_path: 单张图片的本地绝对路径
#     :return: 图片流
#     """
#     img_stream = None
#     with open(img_local_path, 'rb') as img_f:
#         img_stream = img_f.read()
#         img_base64 = str(base64.b64encode(img_stream), encoding='utf-8')
#     return img_base64






if __name__ == "__main__":
    pass