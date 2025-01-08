"""
PatchCore logic based on https://github.com/rvorias/ind_knn_ad
"""
import torch
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import nn

from sklearn import random_projection
from sklearn import linear_model
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from utils_loc.au_pro_util import calculate_au_pro
from timm.models.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_curve
from sklearn.metrics import average_precision_score
from utils_loc.utils import KNNGaussianBlur
from utils_loc.utils import set_seeds
from utils_loc.au_pro_util import calculate_au_pro
import numpy as np
import open3d as o3d
import cv2
from models.pointnet2_utils import interpolating_points
from PIL import Image
from models.models import Model, Mlp
CUDA_LAUNCH_BLOCKING=1
class Features(torch.nn.Module):

    def __init__(self, args, image_size=224, f_coreset=0.1, coreset_eps=0.9):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.deep_feature_extractor = Model(
                                device=self.device, 
                                rgb_backbone_name=args.rgb_backbone_name, 
                                xyz_backbone_name=args.xyz_backbone_name, 
                                group_size = args.group_size, 
                                num_group=args.num_group
                                )
        self.deep_feature_extractor.to(self.device)

        self.args = args
        self.image_size = args.img_size
        self.f_coreset = args.f_coreset
        self.coreset_eps = args.coreset_eps
        
        self.blur = KNNGaussianBlur(4)
        self.n_reweight = 3
        set_seeds(0)
        self.patch_xyz_lib = []
        self.patch_rgb_lib = []
        self.patch_infra_lib = []
        self.patch_fusion_lib = []
        self.patch_lib = []
        self.label_lib = []
        self.random_state = args.random_state

        self.xyz_dim = 0
        self.rgb_dim = 0

        self.xyz_mean=0
        self.xyz_std=0
        self.rgb_mean=0
        self.rgb_std=0
        self.fusion_mean=0
        self.fusion_std=0

        self.average = torch.nn.AvgPool2d(3, stride=1) 
        self.resize = torch.nn.AdaptiveAvgPool2d((56, 56))
        self.resize2 = torch.nn.AdaptiveAvgPool2d((56, 56))
        #--------------------------------------
        # object-level/pixel-level  preds/labels
        # Assign values in multiple_feature.compute_s_s_map() 
        ## object-level
        self.image_preds = list()    
        self.image_labels = list()   
        ## pixel-level
        self.rgb_pixel_preds = list()
        self.rgb_pixel_labels = list()
        self.infra_pixel_preds = list()
        self.infra_pixel_labels = list()
        self.pc_pixel_preds = list()
        self.pc_pixel_labels = list()

        #--------------------------------------
        #pixel-level  preds/labels for predict maps
        #Assign values in multiple_feature.compute_s_s_map() 
        self.rgb_gts = []
        self.rgb_predictions = []
        self.infra_gts = []
        self.infra_predictions = []
        self.pc_gts = []
        self.pc_predictions = []
        self.pc_pts =[]
        #--------------------------------------
        #metrics
        #Assign values in features.calculate_metrics() 
        self.image_rocauc = 0
        self.rgb_pixel_rocauc = 0
        self.rgb_pixel_f1 = 0
        self.rgb_pixel_ap = 0
        self.rgb_pixel_aupr = 0

        self.infra_pixel_rocauc = 0
        self.infra_pixel_f1 = 0
        self.infra_pixel_ap = 0
        self.infra_pixel_aupr = 0

        self.pc_pixel_rocauc = 0
        self.pc_pixel_f1 = 0
        self.pc_pixel_au_pro = 0
        self.pc_pixel_aupr = 0
        #--------------------------------------
        self.au_pro = 0
        self.ins_id = 0
        self.rgb_layernorm = torch.nn.LayerNorm(768, elementwise_affine=False)

        self.detect_fuser = linear_model.SGDOneClassSVM(random_state=42, nu=args.ocsvm_nu,  max_iter=args.ocsvm_maxiter)
        self.seg_fuser = linear_model.SGDOneClassSVM(random_state=42, nu=args.ocsvm_nu,  max_iter=args.ocsvm_maxiter)

      
        self.gate = Mlp(3,128,3)
        self.hard_gate = True
        self.temp = 1.0
        self.loss = nn.modules.loss.BCEWithLogitsLoss()
        self.s_lib = []
        self.s_map_lib = []

    def __call__(self, rgb, infra, xyz):
        # Extract the desired feature maps using the backbone model.
        rgb = rgb.to(self.device)
        infra = infra.to(self.device)
        xyz = xyz.to(self.device)
        with torch.no_grad():
            rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, ori_idx, center_idx ,pts= self.deep_feature_extractor(rgb, infra, xyz)

        interpolate = False  
        if interpolate:
            interpolated_feature_maps = interpolating_points(xyz.float(), center.permute(0,2,1), xyz_feature_maps).to("cpu")

        xyz_feature_maps = [fmap.to("cpu") for fmap in [xyz_feature_maps]]
        rgb_feature_maps = [fmap.to("cpu") for fmap in [rgb_feature_maps]]
        infra_feature_maps = [fmap.to("cpu") for fmap in [infra_feature_maps]]
        if interpolate:
            return rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, ori_idx, center_idx, interpolated_feature_maps
        else:
            return rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, ori_idx, center_idx,pts
        
    def rgb_feats(self, sample):
        rgb = sample[0].to(self.device)
        infra = sample[1].to(self.device)
        xyz = sample[2].to(self.device)
        with torch.no_grad():
            rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, ori_idx, center_idx = self.deep_feature_extractor(rgb, infra, xyz)
        rgb_feature_maps = [fmap.to("cpu") for fmap in [rgb_feature_maps]]
        return rgb_feature_maps
        
    def add_sample_to_mem_bank(self, sample):
        raise NotImplementedError

    def predict(self, sample, mask, label):
        raise NotImplementedError

    def add_sample_to_late_fusion_mem_bank(self, sample):
        raise NotImplementedError

    def interpolate_points(self, rgb, xyz):
        with torch.no_grad():
            rgb_feature_maps, xyz_feature_maps, center, ori_idx, center_idx = self.deep_feature_extractor(rgb, xyz)
        return xyz_feature_maps, center, xyz
    
    def compute_s_s_map(self, xyz_patch, rgb_patch, fusion_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        raise NotImplementedError
    
    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):
        raise NotImplementedError
    
    def run_coreset(self):
        raise NotImplementedError

 

    def calculate_metrics(self):
        
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.expand_dims(np.stack(self.image_labels), axis=1)
        self.rgb_pixel_preds = np.array(self.rgb_pixel_preds)
        self.infra_pixel_preds = np.array(self.infra_pixel_preds)
        self.pc_pixel_preds = np.array(self.pc_pixel_preds)

        self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)

        # metrics for RGB
        if len(self.rgb_pixel_preds) > 0:
            self.rgb_pixel_rocauc = roc_auc_score(self.rgb_pixel_labels, self.rgb_pixel_preds)
            # precision, recall 
            precision, recall, thresholds = precision_recall_curve(self.rgb_pixel_labels, self.rgb_pixel_preds)
            
            # AUPR (Area Under Precision-Recall Curve)
            self.rgb_pixel_aupr = auc(recall, precision)

            # F1 scores
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)  # 避免除零
            
            best_threshold_index = f1_scores.argmax()
            best_threshold = thresholds[best_threshold_index]
            best_f1_score = f1_scores[best_threshold_index]
            
            # F1-max score
            self.rgb_pixel_f1 = best_f1_score
            
            # AP
            self.rgb_pixel_ap = average_precision_score(self.rgb_pixel_labels, self.rgb_pixel_preds)

        # metrics for infra
        if len(self.infra_pixel_preds) > 0:
            self.infra_pixel_rocauc = roc_auc_score(self.infra_pixel_labels, self.infra_pixel_preds)
            precision, recall, thresholds = precision_recall_curve(self.infra_pixel_labels, self.infra_pixel_preds)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
            best_threshold_index = f1_scores.argmax()
            best_threshold = thresholds[best_threshold_index]
            best_f1_score = f1_scores[best_threshold_index]
            
            self.infra_pixel_f1 = best_f1_score
            self.infra_pixel_aupr = auc(recall, precision)
            
            self.infra_pixel_ap = average_precision_score(self.infra_pixel_labels, self.infra_pixel_preds)

         # metrics for point cloud
        if len(self.pc_pixel_preds) > 0:
            self.pc_pixel_rocauc = roc_auc_score(self.pc_pixel_labels, self.pc_pixel_preds)
            precision, recall, thresholds = precision_recall_curve(self.pc_pixel_labels, self.pc_pixel_preds)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
            best_threshold_index = f1_scores.argmax()
            best_threshold = thresholds[best_threshold_index]
            best_f1_score = f1_scores[best_threshold_index]
            
            self.pc_pixel_f1 = best_f1_score
            self.pc_pixel_aupr = auc(recall, precision)
            
   
    def cv2heatmap(self,data):
        colormap = cv2.applyColorMap((data * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return colormap

    def save_prediction_maps(self, output_path, rgb_path, infra_path, xyz_path, save_num=10, mode="rgb"):
        if mode=="rgb":
            for i in range(max(save_num, len(self.rgb_predictions))):                
                fig, axs = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 1]})
            
                ax3 = axs[0]
                gt = Image.open(rgb_path[i][0])
                gt = np.array(gt)  
                ax3.imshow(gt)
                ax3.set_title('Ground Truth')
                ax3.axis('off') 

              
                ax2 = axs[1]
                im2 = ax2.imshow(self.rgb_gts[i], cmap=plt.cm.gray)
                ax2.set_title('Ground Truth (Gray)')
                ax2.axis('off')  

               
                all_predictions = np.concatenate(self.rgb_predictions)
                vmin = all_predictions.min()
                vmax = all_predictions.max()

             
                ax = axs[2]
                im = ax.imshow(self.rgb_predictions[i], cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                ax.set_title('Predictions')
                ax.axis('off') 

          
                plt.colorbar(im, ax=ax)

            
                plt.tight_layout()
                plt.subplots_adjust(wspace=0.4)  
                class_dir = os.path.join(output_path, "RGB")
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir,exist_ok=True)

                ad_dir = os.path.join(class_dir, rgb_path[i][0].split('/')[-2]) 
            
                if not os.path.exists(ad_dir):
                    os.makedirs(ad_dir,exist_ok=True)

                plt.savefig(os.path.join(ad_dir, str(self.image_preds[i]) + '_pred_' + rgb_path[i][0].split('/')[-1] + '.jpg'))
        if mode=="infra":
            for i in range(max(save_num, len(self.infra_predictions))):                
                fig, axs = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 1]})

               
                ax3 = axs[0]
                gt = Image.open(infra_path[i][0])
                gt = np.array(gt)  
                ax3.imshow(gt)
                ax3.set_title('Ground Truth')
                ax3.axis('off') 

             
                ax2 = axs[1]
                im2 = ax2.imshow(self.infra_gts[i], cmap=plt.cm.gray)
                ax2.set_title('Ground Truth (Gray)')
                ax2.axis('off')  

     
                all_predictions = np.concatenate(self.infra_predictions)
                vmin = all_predictions.min()
                vmax = all_predictions.max()

               
                ax = axs[2]
                im = ax.imshow(self.infra_predictions[i], cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
                ax.set_title('Predictions')
                ax.axis('off')  

              
                plt.colorbar(im, ax=ax)

              
                plt.tight_layout()
                plt.subplots_adjust(wspace=0.4)  
                class_dir = os.path.join(output_path, "infra")
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir,exist_ok=True)

                ad_dir = os.path.join(class_dir, infra_path[i][0].split('/')[-2])  
            
                if not os.path.exists(ad_dir):
                    os.makedirs(ad_dir,exist_ok=True)

                plt.savefig(os.path.join(ad_dir, str(self.image_preds[i]) + '_pred_' + infra_path[i][0].split('/')[-1] + '.jpg'))
        
        
        
        if mode=="xyz":
            for i in range(max(save_num, len(self.pc_predictions))): 

                mesh_stl = o3d.geometry.TriangleMesh()
                mesh_stl = o3d.io.read_triangle_mesh(xyz_path[i][0])
                mesh_stl = mesh_stl.remove_duplicated_vertices()
 
                points = np.asarray(mesh_stl.vertices)
                anomaly_scores = np.array(self.pc_predictions[i])

              
                assert points.shape[0] == anomaly_scores.shape[0]

           
                output_data = np.hstack((points, anomaly_scores.reshape(-1, 1)))
                
                class_dir = os.path.join(output_path, "PC")
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir,exist_ok=True)

                ad_dir = os.path.join(class_dir, xyz_path[i][0].split('/')[-2])  
            
                if not os.path.exists(ad_dir):
                    os.makedirs(ad_dir,exist_ok=True)
                
                np.savetxt(
                    os.path.join(ad_dir, str(self.image_preds[i]) + '_pred_' + xyz_path[i][0].split('/')[-1] + '.txt'), 
                    output_data, 
                    fmt='%.10f', 
                    header='X Y Z AnomalyScore', 
                    delimiter=' '
                )


    def run_late_fusion(self):
        
        self.s_lib = torch.cat(self.s_lib, 0)

        self.detect_fuser.fit(self.s_lib)

        

    def get_coreset_idx_randomp(self, z_lib, n=1000, eps=0.90, float16=True, force_cpu=False):

        print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
        try:
            transformer = random_projection.SparseRandomProjection(eps=eps, random_state=self.random_state)
            z_lib = torch.tensor(transformer.fit_transform(z_lib))

            print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
        except ValueError:
            print("   Error: could not project vectors. Please increase `eps`.")

        select_idx = 0
        last_item = z_lib[select_idx:select_idx + 1]
        coreset_idx = [torch.tensor(select_idx)]
        min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)

        if float16:
            last_item = last_item.half()
            z_lib = z_lib.half()
            min_distances = min_distances.half()
        if torch.cuda.is_available() and not force_cpu:
            last_item = last_item.to("cuda")
            z_lib = z_lib.to("cuda")
            min_distances = min_distances.to("cuda")

        for _ in tqdm(range(n - 1)):
            distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)  # broadcasting step
            min_distances = torch.minimum(distances, min_distances)  # iterative step
            select_idx = torch.argmax(min_distances)  # selection step

            # bookkeeping
            last_item = z_lib[select_idx:select_idx + 1]
            min_distances[select_idx] = 0
            coreset_idx.append(select_idx.to("cpu"))
        return torch.stack(coreset_idx)
