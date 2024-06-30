import torch
from feature_extractors.features import Features
from utils_loc.mvtec3d_util import *
import numpy as np
import math
import os
import random

##Single modality
class RGBFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        # self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)
        # self.patch_infra_lib.append(infra_patch)
        
    def predict(self, sample, label):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
 
            
        

        # 2D-true dist 
        # xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        # infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        # dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        # dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        # infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        # xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        # s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        # s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
        

        
        s = torch.tensor([s_rgb])
        
        # s_map = torch.cat([10*s_map_xyz, s_map_rgb, s_map_infra], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)

        # self.s_map_lib.append(s_map)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        # xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        # infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        # dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        # dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        # infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        # xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
        # s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        # s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([s_rgb])
        # s = torch.tensor(self.detect_fuser.score_samples(s))

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
 

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_infra_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        # self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        # self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)
        
        # self.xyz_mean = torch.mean(self.patch_xyz_lib)
        # self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        # self.infra_mean = torch.mean(self.patch_xyz_lib)
        # self.infra_std = torch.std(self.patch_rgb_lib)
        
        # self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        
        # self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:
            # self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
            #                                                 n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
            #                                                 eps=self.coreset_eps, )
            # self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            # self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
            #                                                 n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
            #                                                 eps=self.coreset_eps, )
            # self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx] 
class InfraFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        # self.patch_xyz_lib.append(xyz_patch)
        # self.patch_rgb_lib.append(rgb_patch)
        self.patch_infra_lib.append(infra_patch)
        
    def predict(self, sample, label):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
 
            
        

        # 2D-true dist 
        # xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        # rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        # dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        # dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        # rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        # xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        # s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        # s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
        

        
        s = torch.tensor([s_infra])
        

        self.s_lib.append(s)


    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        # xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        # rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        # dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        # dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        # rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        # xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
        # s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        # s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([s_infra])
        # s = torch.tensor(self.detect_fuser.score_samples(s))

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
 

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_infra_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        # self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        # self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)
        
        # self.xyz_mean = torch.mean(self.patch_xyz_lib)
        # self.xyz_std = torch.std(self.patch_rgb_lib)
        # self.rgb_mean = torch.mean(self.patch_xyz_lib)
        # self.rgb_std = torch.std(self.patch_rgb_lib)
        self.infra_mean = torch.mean(self.patch_infra_lib)
        self.infra_std = torch.std(self.patch_infra_lib)
        
        # self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        # self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:
            # self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
            #                                                 n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
            #                                                 eps=self.coreset_eps, )
            # self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            # self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
            #                                                 n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
            #                                                 eps=self.coreset_eps, )
            # self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
                                                            n=int(self.f_coreset * self.patch_infra_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]  
class PCFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        self.patch_xyz_lib.append(xyz_patch)
        # self.patch_rgb_lib.append(rgb_patch)
        # self.patch_infra_lib.append(infra_patch)
        
    def predict(self, sample, label):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
 
            
        

        # 2D-true dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        # rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        # infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        # dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        # dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        # infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        # rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        # s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        # s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
             
        s = torch.tensor([s_xyz])
        
        self.s_lib.append(s)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        # rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        # infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        # dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        # dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        # infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        # rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        # s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        # s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([s_xyz])

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
 

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_infra_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        # self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        # self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)
        
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_xyz_lib)
        # self.rgb_mean = torch.mean(self.patch_xyz_lib)
        # self.rgb_std = torch.std(self.patch_rgb_lib)
        # self.infra_mean = torch.mean(self.patch_xyz_lib)
        # self.infra_std = torch.std(self.patch_rgb_lib)
        
        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        # self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        
        # self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            # self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
            #                                                 n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
            #                                                 eps=self.coreset_eps, )
            # self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            # self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
            #                                                 n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
            #                                                 eps=self.coreset_eps, )
            # self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]  
## Decision level fusion
# with raw add
class PCRGBAddFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)
        # self.patch_infra_lib.append(infra_patch)
        
    def predict(self, sample, label):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
 
            
        

        # 2D-true dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        # infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        # dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        # infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        # s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
        

        
        s = torch.tensor([(s_xyz+s_rgb*0.1)/2])
        
        # s_map = torch.cat([10*s_map_xyz, s_map_rgb, s_map_infra], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)

        # self.s_map_lib.append(s_map)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        # infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        # dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        # infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        # s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([(s_xyz+s_rgb*0.1)/2])

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
 

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_infra_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        # self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)
        
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        # self.infra_mean = torch.mean(self.patch_xyz_lib)
        # self.infra_std = torch.std(self.patch_rgb_lib)
        
        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        
        # self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            # self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
            #                                                 n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
            #                                                 eps=self.coreset_eps, )
            # self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]  
class PCInfraAddFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        self.patch_xyz_lib.append(xyz_patch)
        # self.patch_rgb_lib.append(rgb_patch)
        self.patch_infra_lib.append(infra_patch)
        
    def predict(self, sample, label):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        # rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        # rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
 
            
        

        # 2D-true dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        # rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        # dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        # rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        # s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
        

        
        s = torch.tensor([(s_xyz+s_infra)/2])
        
        # s_map = torch.cat([10*s_map_xyz, s_map_rgb, s_map_infra], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)

        # self.s_map_lib.append(s_map)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        # rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        # dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        # rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        # s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([(s_xyz+s_infra)/2])
        # s = torch.tensor(self.detect_fuser.score_samples(s))

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
 

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_infra_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        # self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)
        
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_infra_lib)
        # self.rgb_mean = torch.mean(self.patch_rgb_lib)
        # self.rgb_std = torch.std(self.patch_rgb_lib)
        self.infra_mean = torch.mean(self.patch_xyz_lib)
        self.infra_std = torch.std(self.patch_infra_lib)
        
        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        # self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            # self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
            #                                                 n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
            #                                                 eps=self.coreset_eps, )
            # self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
                                                            n=int(self.f_coreset * self.patch_infra_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]               
class RGBInfraAddFeatures(Features):
    
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(sample[0], sample[1], sample[2])

        # xyz_patch = torch.cat(xyz_feature_maps, 1)
        # xyz_patch = xyz_patch.squeeze(0).T
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        # self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)
        self.patch_infra_lib.append(infra_patch)
        
    def predict(self, sample, label):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
 
            
        

        # 2D-true dist 
        # xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        # dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        # xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        # s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
        

        
        s = torch.tensor([(s_rgb+s_infra)/2])
        

        self.s_lib.append(s)

        # self.s_map_lib.append(s_map)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        # xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        # dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        # xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
        # s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([(s_rgb+s_infra)/2])

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
 

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_infra_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        # self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)
        
        # self.xyz_mean = torch.mean(self.patch_xyz_lib)
        # self.xyz_std = torch.std(self.patch_xyz_lib)
        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_infra_lib)
        self.infra_mean = torch.mean(self.patch_rgb_lib)
        self.infra_std = torch.std(self.patch_infra_lib)
        
        # self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:
            # self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
            #                                                 n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
            #                                                 eps=self.coreset_eps, )
            # self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
                                                            n=int(self.f_coreset * self.patch_infra_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]
class TripleAddFeatures(Features):
    
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)
        self.patch_infra_lib.append(infra_patch)
        
    def predict(self, sample, label):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
 
            
        

        # 2D-true dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
        

        
        s = torch.tensor([(s_xyz+s_rgb+s_infra)/3])
        

        self.s_lib.append(s)

        # self.s_map_lib.append(s_map)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([(s_xyz+s_rgb+s_infra)/3])

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
 

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_infra_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)
        
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.infra_mean = torch.mean(self.patch_xyz_lib)
        self.infra_std = torch.std(self.patch_rgb_lib)
        
        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
                                                            n=int(self.f_coreset * self.patch_infra_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]
# with gating 
class PCRGBGatingFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)
        # self.patch_infra_lib.append(infra_patch)
        
    def predict(self, sample, label):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
 
            
        

        # 2D-true dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        # infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        # dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        # infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        # s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
        

        
        s = torch.tensor([[s_xyz, s_rgb]])
        
        # s_map = torch.cat([10*s_map_xyz, s_map_rgb, s_map_infra], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)


    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        # infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        # dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        # infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        # s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([[s_xyz, s_rgb]])
        s = torch.tensor(self.detect_fuser.score_samples(s))

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
 

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_infra_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        # self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)
        
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        # self.infra_mean = torch.mean(self.patch_xyz_lib)
        # self.infra_std = torch.std(self.patch_rgb_lib)
        
        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        
        # self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            # self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
            #                                                 n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
            #                                                 eps=self.coreset_eps, )
            # self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]
class PCInfraGatingFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)

        xyz_patch = xyz_patch.squeeze(0).T
        

        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_infra_lib.append(infra_patch)
        

        
    def predict(self, sample, label):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        # rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        # rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        

            
        

        # 2D-true dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        # rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        # dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        # rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        # s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
        
        
        s = torch.tensor([[s_xyz, s_infra]])
        

        self.s_lib.append(s)
        # self.s_map_lib.append(s_map)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        # rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        # dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        # rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        # s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([[s_xyz, s_infra]])
 

        
        s = torch.tensor(self.detect_fuser.score_samples(s))



        self.image_preds.append(s.numpy())
        self.image_labels.append(label)

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_infra_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        # self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)
        
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_infra_lib)
        # self.rgb_mean = torch.mean(self.patch_rgb_lib)
        # self.rgb_std = torch.std(self.patch_rgb_lib)
        self.infra_mean = torch.mean(self.patch_xyz_lib)
        self.infra_std = torch.std(self.patch_infra_lib)
        
        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        # self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            # self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
            #                                                 n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
            #                                                 eps=self.coreset_eps, )
            # self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
                                                            n=int(self.f_coreset * self.patch_infra_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]
class RGBInfraGatingFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(sample[0], sample[1], sample[2])
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        self.patch_rgb_lib.append(rgb_patch)
        self.patch_infra_lib.append(infra_patch)

        
    def predict(self, sample, label):
 
        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
 

        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        # 2D-true dist 
        # xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        # dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        # xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        # s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
          
        s = torch.tensor([[s_rgb, s_infra]])

        self.s_lib.append(s)
 

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        # xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        # dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        # xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
        # s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([[s_rgb, s_infra]])
        # s_map = torch.cat([self.args.xyz_smap_lambda*s_map_xyz, self.args.rgb_smap_lambda*s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)

        
        s = torch.tensor(self.detect_fuser.score_samples(s))

        # s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
        # s_map = s_map.view(1, 224, 224)
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_infra_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        
        # self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)
        
        # self.xyz_mean = torch.mean(self.patch_xyz_lib)
        # self.xyz_std = torch.std(self.patch_xyz_lib)
        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_infra_lib)
        self.infra_mean = torch.mean(self.patch_rgb_lib)
        self.infra_std = torch.std(self.patch_infra_lib)
        
        # self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:
            # self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
            #                                                 n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
            #                                                 eps=self.coreset_eps, )
            # self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
                                                            n=int(self.f_coreset * self.patch_infra_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]
class TripleRGBInfraPointFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)

        xyz_patch = xyz_patch.squeeze(0).T
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        # rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)
        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T


        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)
        self.patch_infra_lib.append(infra_patch)
        
    def noise_gen(self, feat):
        noise_std = 0.015
        noise_idxs = torch.randint(0, 1, torch.Size([feat.shape[0]]))
        noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=1).to(self.device) # (N, K)
        noise = torch.stack([
                torch.normal(0, noise_std * 1.1**(k), feat.shape)
                for k in range(1)], dim=1).to(self.device) # (N, K, C)
        noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
        return noise
    
    def code_gen(self):
        while True:
            array = [random.choice([0,1]) for _ in range(3)]
            if array != [0,0,0]:
                return array
        
    def predict(self, sample, label):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        

            
        

        # 2D-true dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
        
 
        
        
        

        
        s = torch.tensor([[s_xyz, s_rgb, s_infra]])

        
        s_map = torch.cat([10*s_map_xyz, s_map_rgb, s_map_infra], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)
 
        self.s_map_lib.append(s_map)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([[s_xyz, s_rgb, s_infra]])
        s = torch.tensor(self.detect_fuser.score_samples(s))

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)


    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_infra_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)
        
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        # self.xyz_std = torch.std(self.patch_xyz_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        # self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.infra_mean = torch.mean(self.patch_xyz_lib)
        # self.infra_mean = torch.mean(self.patch_infra_lib)
        self.infra_std = torch.std(self.patch_rgb_lib)
        
        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]
            



