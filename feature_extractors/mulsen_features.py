import torch
from feature_extractors.features import Features
from utils_loc.mvtec3d_util import *
import numpy as np
import math
import os
import random
from utils_loc.cpu_knn import fill_missing_values
##Single modality
class RGBFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(sample[0], sample[1], sample[2])
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        self.patch_rgb_lib.append(rgb_patch)
        
    def predict(self, sample, label, pixel_mask):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        

        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s,pixel_mask)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
 

        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()

        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)



   
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
  

        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
      
        

        
        s = torch.tensor([s_rgb])
        


        self.s_lib.append(s)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label,pixel_mask):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

    
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
 
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
  

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))

        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')


        s = torch.tensor([s_rgb])

        #--------------------------------------------------------------
        # object-level preds  or labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)

        # pixel-level preds  or labels
        ## RGB
        self.rgb_pixel_preds.extend(s_map_rgb.flatten().numpy())
        self.rgb_pixel_labels.extend(pixel_mask[0].flatten().numpy())
        
        self.rgb_predictions.append(s_map_rgb.detach().cpu().squeeze().numpy())
        self.rgb_gts.append(pixel_mask[0].detach().cpu().squeeze().numpy())

        ## Infra

        ## PC
     
        #-------------------------------------------------------------

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):
     
        
        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0) 

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0) 
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0) 
            w_dist = torch.cdist(m_star, self.patch_infra_lib)  

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)   

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
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear',align_corners=True)
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
      
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
      

        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)


        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        

        if self.f_coreset < 1:

            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

            
class InfraFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        # self.patch_xyz_lib.append(xyz_patch)
        # self.patch_rgb_lib.append(rgb_patch)
        self.patch_infra_lib.append(infra_patch)
        
    def predict(self, sample, label, pixel_mask):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s, pixel_mask)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        

        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()

        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))

        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
        

        
        s = torch.tensor([s_infra])
        

        self.s_lib.append(s)


    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label,pixel_mask):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()

        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))

        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([s_infra])
 
        #--------------------------------------------------------------
        # object-level preds  or labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
         # pixel-level preds  or labels
        ## RGB


        ## Infra
        self.infra_pixel_preds.extend(s_map_infra.flatten().numpy())
        self.infra_pixel_labels.extend(pixel_mask[1].flatten().numpy())
        
        self.infra_predictions.append(s_map_infra.detach().cpu().squeeze().numpy())
        self.infra_gts.append(pixel_mask[1].detach().cpu().squeeze().numpy())
        ## PC

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)   

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0) 
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)   
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_infra_lib)   

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)   

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
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear',align_corners=True)
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):

        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)

        self.infra_mean = torch.mean(self.patch_infra_lib)
        self.infra_std = torch.std(self.patch_infra_lib)

        
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:

            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
                                                            n=int(self.f_coreset * self.patch_infra_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]  

class PCFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T  
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.patch_xyz_lib.append(xyz_patch)

        
    def predict(self, sample, label, pixel_mask):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,pts = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s,pixel_mask,center_idx,pts)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,pts = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        # 2D-true dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)

        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
    
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size,center_idx,pts, modal='xyz')

             
        s = torch.tensor([s_xyz])
        
        self.s_lib.append(s)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label,pixel_mask,center_idx,pts):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)

        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size,center_idx,pts, modal='xyz')


        s = torch.tensor([s_xyz])

        #--------------------------------------------------------------
        # object-level preds  or labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        # pixel-level preds  or labels
        ## RGB


        ## Infra

        ## PC
        self.pc_pixel_preds.extend(s_map_xyz.flatten())
        self.pc_pixel_labels.extend(pixel_mask[2].flatten().numpy())
        
        self.pc_pts.append(pts[0,:])
        self.pc_predictions.append(s_map_xyz.squeeze())
        self.pc_gts.append(pixel_mask[2].detach().cpu().squeeze().numpy())
        #-------------------------------------------------------------

    def compute_single_s_s_map(self, patch, dist, feature_map_dims,center_idx,pts, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)   

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)  
            w_dist = torch.cdist(m_star, self.patch_infra_lib)  

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)   

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        if modal=='xyz':
            s_xyz_map = min_val
            if not center_idx.dtype == torch.long:
                center_idx = center_idx.long()

            sample_data = pts[0,center_idx]
            s_xyz_map = s_xyz_map.cpu().numpy()
            full_s_xyz_map = fill_missing_values(sample_data,s_xyz_map,pts, k=1)
            
            return s, full_s_xyz_map


        

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)      
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_xyz_lib)
        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std


        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
 


## Decision level fusion
class PCRGBGatingFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)

        
    def predict(self, sample, label,pixel_mask):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s, pixel_mask, center_idx, pts)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
   
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
 



        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size,center_idx, pts, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
    
        

        
        s = torch.tensor([[s_xyz, s_rgb]])

        self.s_lib.append(s)


    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label, pixel_mask, center_idx, pts):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

      
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
       
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size,center_idx,pts, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[s_xyz, s_rgb]])
        s = torch.tensor(self.detect_fuser.score_samples(s))

        #--------------------------------------------------------------
        # object-level preds  or labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)

        # pixel-level preds  or labels
        ## RGB
        self.rgb_pixel_preds.extend(s_map_rgb.flatten().numpy())
        self.rgb_pixel_labels.extend(pixel_mask[0].flatten().numpy())
        
        self.rgb_predictions.append(s_map_rgb.detach().cpu().squeeze().numpy())
        self.rgb_gts.append(pixel_mask[0].detach().cpu().squeeze().numpy())

        ## Infra

        ## PC
        self.pc_pixel_preds.extend(s_map_xyz.flatten())
        self.pc_pixel_labels.extend(pixel_mask[2].flatten().numpy())
        
        self.pc_pts.append(pts[0,:])
        self.pc_predictions.append(s_map_xyz.squeeze())
        self.pc_gts.append(pixel_mask[2].detach().cpu().squeeze().numpy())

        #-------------------------------------------------------------
 

    def compute_single_s_s_map(self, patch, dist, feature_map_dims,center_idx=None, pts=None, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

   
        m_test = patch[s_idx].unsqueeze(0)   

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)   
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)   
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_infra_lib)   

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)   

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        if modal=="xyz": 
            s_xyz_map = min_val
            if not center_idx.dtype == torch.long:
                center_idx = center_idx.long()

            sample_data = pts[0,center_idx]
            s_xyz_map = s_xyz_map.cpu().numpy()
            full_s_xyz_map = fill_missing_values(sample_data,s_xyz_map,pts, k=1)
            
            return s, full_s_xyz_map
        else:
            s_map = min_val.view(1, 1, *feature_map_dims)
            s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear',align_corners=True)
            s_map = self.blur(s_map)

            return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)

        
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)

        
        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

class PCInfraGatingFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, _= self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)

        xyz_patch = xyz_patch.squeeze(0).T
        

        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_infra_lib.append(infra_patch)
        

        
    def predict(self, sample, label, pixel_mask):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s, pixel_mask, center_idx, pts)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        

            
        

        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

    
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size,center_idx,pts, modal='xyz')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
        
        
        s = torch.tensor([[s_xyz, s_infra]])
        

        self.s_lib.append(s)


    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label, pixel_mask, center_idx, pts):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

    
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()

        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)

        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))

        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
       
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size,center_idx,pts, modal='xyz')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([[s_xyz, s_infra]])
 

        
        s = torch.tensor(self.detect_fuser.score_samples(s))

        #--------------------------------------------------------------
        # object-level preds  or labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)

        # pixel-level preds  or labels
        ## RGB

        ## Infra
        self.infra_pixel_preds.extend(s_map_infra.flatten().numpy())
        self.infra_pixel_labels.extend(pixel_mask[1].flatten().numpy())
        
        self.infra_predictions.append(s_map_infra.detach().cpu().squeeze().numpy())
        self.infra_gts.append(pixel_mask[1].detach().cpu().squeeze().numpy())

        ## PC
        self.pc_pixel_preds.extend(s_map_xyz.flatten())
        self.pc_pixel_labels.extend(pixel_mask[2].flatten().numpy())
        
        self.pc_pts.append(pts[0,:])
        self.pc_predictions.append(s_map_xyz.squeeze())
        self.pc_gts.append(pixel_mask[2].detach().cpu().squeeze().numpy())

        #-------------------------------------------------------------

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, center_idx=None, pts=None, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000


        m_test = patch[s_idx].unsqueeze(0)   

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)   
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)   
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_infra_lib)   

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)   

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        if modal=="xyz": 
            s_xyz_map = min_val
            if not center_idx.dtype == torch.long:
                center_idx = center_idx.long()

            sample_data = pts[0,center_idx]
            s_xyz_map = s_xyz_map.cpu().numpy()
            full_s_xyz_map = fill_missing_values(sample_data,s_xyz_map,pts, k=1)
            
            return s, full_s_xyz_map
        else:
            # segmentation map
            s_map = min_val.view(1, 1, *feature_map_dims)
            s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear',align_corners=True)
            s_map = self.blur(s_map)

            return s, s_map

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)

        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)
        
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_infra_lib)

        self.infra_mean = torch.mean(self.patch_xyz_lib)
        self.infra_std = torch.std(self.patch_infra_lib)
        
        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std


        
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]

            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
                                                            n=int(self.f_coreset * self.patch_infra_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]

class RGBInfraGatingFeatures(Features):

    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, _ = self(sample[0], sample[1], sample[2])
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        self.patch_rgb_lib.append(rgb_patch)
        self.patch_infra_lib.append(infra_patch)

        
    def predict(self, sample, label, pixel_mask):
 
        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(rgb, infra, pc)
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s, pixel_mask, center_idx, pts)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, _ = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
 

        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        

        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
   
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))

        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
          
        s = torch.tensor([[s_rgb, s_infra]])

        self.s_lib.append(s)
 

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label, pixel_mask, center_idx, pts):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''


        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()

        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))

        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([[s_rgb, s_infra]])

        s = torch.tensor(self.detect_fuser.score_samples(s))
        #--------------------------------------------------------------
        # object-level preds  or labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)

        # pixel-level preds  or labels
        ## RGB
        self.rgb_pixel_preds.extend(s_map_rgb.flatten().numpy())
        self.rgb_pixel_labels.extend(pixel_mask[0].flatten().numpy())
        
        self.rgb_predictions.append(s_map_rgb.detach().cpu().squeeze().numpy())
        self.rgb_gts.append(pixel_mask[0].detach().cpu().squeeze().numpy())

        ## Infra
        self.infra_pixel_preds.extend(s_map_infra.flatten().numpy())
        self.infra_pixel_labels.extend(pixel_mask[1].flatten().numpy())
        
        self.infra_predictions.append(s_map_infra.detach().cpu().squeeze().numpy())
        self.infra_gts.append(pixel_mask[1].detach().cpu().squeeze().numpy())

        ## PC
    
        #-------------------------------------------------------------

    def compute_single_s_s_map(self, patch, dist, feature_map_dims,center_idx=None, pts=None, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000


        m_test = patch[s_idx].unsqueeze(0)   

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)   
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)   
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_infra_lib)   

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)   

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        if modal=="xyz": 
            s_xyz_map = min_val
            if not center_idx.dtype == torch.long:
                center_idx = center_idx.long()

            sample_data = pts[0,center_idx]
            s_xyz_map = s_xyz_map.cpu().numpy()
            full_s_xyz_map = fill_missing_values(sample_data,s_xyz_map,pts, k=1)
            
            return s, full_s_xyz_map
        else:
            # segmentation map
            s_map = min_val.view(1, 1, *feature_map_dims)
            s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear',align_corners=True)
            s_map = self.blur(s_map)

            return s, s_map

    def run_coreset(self):
        

        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)
        

        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_infra_lib)
        self.infra_mean = torch.mean(self.patch_rgb_lib)
        self.infra_std = torch.std(self.patch_infra_lib)
        

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:

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

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, _ = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)

        xyz_patch = xyz_patch.squeeze(0).T
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T


        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T


        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)
        self.patch_infra_lib.append(infra_patch)
        
        
    def predict(self, sample, label,pixel_mask):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(rgb, infra, pc)
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s, pixel_mask, center_idx, pts)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size,center_idx, pts, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([[s_xyz, s_rgb, s_infra]])


        self.s_lib.append(s)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label, pixel_mask,center_idx,pts):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size,center_idx, pts,modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([[s_xyz, s_rgb, s_infra]])
        s = torch.tensor(self.detect_fuser.score_samples(s))

        #--------------------------------------------------------------
        # object-level preds  or labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)

        # pixel-level preds  or labels
        ## RGB
        self.rgb_pixel_preds.extend(s_map_rgb.flatten().numpy())
        self.rgb_pixel_labels.extend(pixel_mask[0].flatten().numpy())
        
        self.rgb_predictions.append(s_map_rgb.detach().cpu().squeeze().numpy())
        self.rgb_gts.append(pixel_mask[0].detach().cpu().squeeze().numpy())

        ## Infra
        self.infra_pixel_preds.extend(s_map_infra.flatten().numpy())
        self.infra_pixel_labels.extend(pixel_mask[1].flatten().numpy())
        
        self.infra_predictions.append(s_map_infra.detach().cpu().squeeze().numpy())
        self.infra_gts.append(pixel_mask[1].detach().cpu().squeeze().numpy())

        ## PC
        self.pc_pixel_preds.extend(s_map_xyz.flatten())
        self.pc_pixel_labels.extend(pixel_mask[2].flatten().numpy())
        
        self.pc_pts.append(pts[0,:])
        self.pc_predictions.append(s_map_xyz.squeeze())
        self.pc_gts.append(pixel_mask[2].detach().cpu().squeeze().numpy())


        #-------------------------------------------------------------

    def compute_single_s_s_map(self, patch, dist, feature_map_dims,center_idx=None, pts=None, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)   

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)   
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)   
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_infra_lib)   

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)   

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        if modal=="xyz": 
            s_xyz_map = min_val
            if not center_idx.dtype == torch.long:
                center_idx = center_idx.long()

            sample_data = pts[0,center_idx]
            s_xyz_map = s_xyz_map.cpu().numpy()
            full_s_xyz_map = fill_missing_values(sample_data,s_xyz_map,pts, k=1)
            
            return s, full_s_xyz_map
        else:
            # segmentation map
            s_map = min_val.view(1, 1, *feature_map_dims)
            s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear',align_corners=True)
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
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]
            



