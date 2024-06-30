import open3d as o3d
import numpy as np
import os
import glob
from PIL import Image

def bmp2png(file_path):
    
    if file_path.endswith('.bmp'):
        target_path = file_path.replace('bmp', 'png')
        with Image.open(file_path) as img:
            img.save(target_path, 'PNG')
            print("转换完成，文件已保存为:", target_path)
# if __name__ == '__main__':

#     mulsen = [
#         "capsule",
#         "cube",
#         "cotton",
#             ]
#     split = [
#         'train',
#         'test',
#             ]
#     modality = [
#         'RGB',
#         'Infrared',
#     ]
#     for cls in mulsen:
#         # for modal
#         for sp in split:
#             bmp_list = sorted(glob.glob('/home/dataset/liwq/datasets/mulsen_datasets/{}/{}/train/*.bmp'.format(cls),.format{}))
#             for bmp_file in bmp_list:
#                 bmp2png(bmp_file)