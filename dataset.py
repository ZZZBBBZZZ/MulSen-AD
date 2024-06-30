import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils_loc.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np
import trimesh
import open3d as o3d
import csv
import re


def mulsen_classes():
    return [
        "capsule",
        "cotton",
        "cube",
        "spring_pad",
        "screw",
        "screen",
        "piggy",
        "nut",
        "flat_pad",
        'plastic_cylinder',
        "zipper",
        "button_cell",
        "toothbrush",
        "solar_panel",
        "light",
    ]

RGB_SIZE = 224

class BaseAnomalyDetectionDataset(Dataset):

    def __init__(self, split, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(dataset_path, self.cls)
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])
        
        ##self.pc_transform = ...
        
    def sort(self, file_paths):
        
        paths_with_numbers = []
        pattern = re.compile(r'(\d+)\.(png|stl)$')
        for path in file_paths:
            match = pattern.search(path)
            if match:
                number = int(match.group(1))
                paths_with_numbers.append((path, number))
        paths_with_numbers.sort(key=lambda x: x[1])    
        sorted_paths = [p[0] for p in paths_with_numbers]
        return sorted_paths
    
class PreTrainTensorDataset(Dataset):
    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path
        self.tensor_paths = os.listdir(self.root_path)


    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]

        tensor = torch.load(os.path.join(self.root_path, tensor_path))

        label = 0

        return tensor, label
class PreTrainTensorDataset(Dataset):
    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path
        self.tensor_paths = os.listdir(self.root_path)


    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]

        tensor = torch.load(os.path.join(self.root_path, tensor_path))

        label = 0

        return tensor, label

class TrainDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        super().__init__(split="train", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        rgb_paths = glob.glob(os.path.join(self.img_path, 'RGB', 'train') + "/*.png")
        infra_paths = glob.glob(os.path.join(self.img_path, 'Infrared', 'train') + "/*.png")
        pc_paths = glob.glob(os.path.join(self.img_path, 'Pointcloud', 'train') + "/*.stl")
        
        # tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        rgb_paths = self.sort(rgb_paths)
        infra_paths = self.sort(infra_paths)
        pc_paths = self.sort(pc_paths)
        sample_paths = list(zip(rgb_paths, infra_paths, pc_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels
    def norm_pcd(self, point_cloud):

        center = np.average(point_cloud,axis=0)
        # print(center.shape)
        new_points = point_cloud-np.expand_dims(center,axis=0)
        return new_points
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        infra_path = img_path[1]
        pc_path = img_path[2]
        img = Image.open(rgb_path).convert('RGB')
        infra = Image.open(infra_path).convert('RGB')
        img = self.rgb_transform(img)
        infra = self.rgb_transform(infra)
        # organized_pc = read_tiff_organized_pc(tiff_path)
        mesh_stl = o3d.geometry.TriangleMesh()
        mesh_stl = o3d.io.read_triangle_mesh(pc_path)
        # 合并重复顶点
        mesh_stl = mesh_stl.remove_duplicated_vertices()
        pc = np.asarray(mesh_stl.vertices)
        N = pc.shape[0]
        # if N < 16384:
        #     print(pc_path)
        #     print(N)
        #     up_pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh_stl,number_of_points=16384)
        #     pc = np.array(up_pcd.points)
        pointcloud = self.norm_pcd(pc)
        pointcloud = pointcloud.T
        # depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        # resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        # resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.size, target_width=self.size)
        # resized_organized_pc = resized_organized_pc.clone().detach().float()

        return (img, infra, pointcloud), label


class TestDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        super().__init__(split="test", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.gt_transform = transforms.Compose([
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        # self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.img_paths, self.labels = self.load_dataset() 
        
    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        label_rgb = []
        label_infra = []
        label_pc = []
        
        defect_types = os.listdir(os.path.join(self.img_path, 'RGB', 'test'))
        

        for defect_type in defect_types:
            label_rgb = []
            label_infra = []
            label_pc = []
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, 'RGB', 'test', defect_type) + "/*.png")
                infra_paths = glob.glob(os.path.join(self.img_path, 'Infrared', 'test', defect_type) + "/*.png")
                pc_paths = glob.glob(os.path.join(self.img_path, 'Pointcloud', 'test', defect_type) + "/*.stl")

                        
                rgb_paths = self.sort(rgb_paths)
                infra_paths = self.sort(infra_paths)
                pc_paths = self.sort(pc_paths)
                
                sample_paths = list(zip(rgb_paths, infra_paths, pc_paths))
                img_tot_paths.extend(sample_paths)
                
                # gt_tot_paths.extend([0] * len(sample_paths))
                # tot_labels.extend([0,0,0] * len(sample_paths))
                label_rgb.extend([0] * len(sample_paths))
                label_infra.extend([0] * len(sample_paths))
                label_pc.extend([0] * len(sample_paths))
                label = list(zip(label_rgb, label_infra, label_pc))
                tot_labels.extend(label)
            else:
                with open(os.path.join(self.img_path,'RGB','GT',defect_type,'data.csv'),'r') as file:
                    csvreader = csv.reader(file)
                    header = next(csvreader)
                    for row in csvreader:
                        
                        object, label1, label2, label3 = row
                        label_rgb.extend([int(label1)])
                        label_infra.extend([int(label2)])
                        label_pc.extend([int(label3)])
                label = list(zip(label_rgb, label_infra, label_pc))
                tot_labels.extend(label)
                rgb_paths = glob.glob(os.path.join(self.img_path, 'RGB', 'test', defect_type) + "/*.png")
                infra_paths = glob.glob(os.path.join(self.img_path, 'Infrared', 'test', defect_type) + "/*.png")
                pc_paths = glob.glob(os.path.join(self.img_path, 'Pointcloud', 'test', defect_type) + "/*.stl")
                # gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths = self.sort(rgb_paths)
                infra_paths = self.sort(infra_paths)
                pc_paths = self.sort(pc_paths)
                sample_paths = list(zip(rgb_paths, infra_paths, pc_paths))

                img_tot_paths.extend(sample_paths)
                # gt_tot_paths.extend(gt_paths)
                # tot_labels.extend([1] * len(sample_paths))

        assert len(img_tot_paths) == len(tot_labels), "Something wrong with test and ground truth pair!"

        # return img_tot_paths, gt_tot_paths, tot_labels
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)
    
    def norm_pcd(self, point_cloud):

        center = np.average(point_cloud,axis=0)
        # print(center.shape)
        new_points = point_cloud-np.expand_dims(center,axis=0)
        return new_points
    
    
    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        infra_path = img_path[1]
        pc_path = img_path[2]
        img = Image.open(rgb_path).convert('RGB')
        infra = Image.open(infra_path).convert('RGB')
        img = self.rgb_transform(img)
        infra = self.rgb_transform(infra)
        mesh_stl = o3d.geometry.TriangleMesh()
        mesh_stl = o3d.io.read_triangle_mesh(pc_path)
        mesh_stl = mesh_stl.remove_duplicated_vertices()
        pc = np.asarray(mesh_stl.vertices)
        N = pc.shape[0]
        pointcloud = self.norm_pcd(pc)
        pointcloud = pointcloud.T
        

        return (img, infra, pointcloud), label, rgb_path


def get_data_loader(split, class_name, img_size, args):
    if split in ['train']:
        dataset = TrainDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path)
    elif split in ['test']:
        dataset = TestDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path)

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                             pin_memory=True)
    return data_loader


