import torch
from tqdm import tqdm
import os
from utils_loc import misc
from feature_extractors import mulsen_features
import pandas as pd
from dataset import get_data_loader
from models.models import Model, Mlp
import time
class Tester():
    def __init__(self, args):
        self.args = args
        self.image_size = args.img_size
        if args.method_name == 'RGB':
            self.methods = {
                "RGB": mulsen_features.RGBFeatures(args),
            }
        elif args.method_name == 'Infra':
            self.methods = {
                "Infra": mulsen_features.InfraFeatures(args),
            }
        elif args.method_name == 'PC':
            self.methods = {
                "PC": mulsen_features.PCFeatures(args),
            }
        elif args.method_name == 'PC+RGB+gating':
            self.methods = {
                "PC+RGB+gating": mulsen_features.PCRGBGatingFeatures(args),
            }
        elif args.method_name == 'PC+Infra+gating':
            self.methods = {
                "PC+Infra+gating": mulsen_features.PCInfraGatingFeatures(args),
            }
        elif args.method_name == 'RGB+Infra+gating':
            self.methods = {
                "RGB+Infra+gating": mulsen_features.RGBInfraGatingFeatures(args),
            }
        elif args.method_name == 'PC+RGB+Infra+gating':
            self.methods = {
                "PC+RGB+Infra+gating": mulsen_features.TripleRGBInfraPointFeatures(args),
            }


    def fit(self, class_name):
        train_loader = get_data_loader("train", class_name=class_name, img_size=self.image_size, args=self.args)


        print(f'Extracting train features for class {class_name}')
        for sample, _ in train_loader:
            for method in self.methods.values():
                if self.args.save_feature:
                    method.add_sample_to_mem_bank(sample, class_name=class_name)
                else:
                    method.add_sample_to_mem_bank(sample)
    

        for method_name, method in self.methods.items():
            print(f'\n\nRunning coreset for {method_name} on class {class_name}...')
            method.run_coreset()
            
        if self.args.memory_bank == 'multiple':    
         
            print(f'MultiScoring for {method_name} on class {class_name}..')
            for sample, _ in train_loader: 
                for method_name, method in self.methods.items():
                    method.add_sample_to_late_fusion_mem_bank(sample)
      

            for method_name, method in self.methods.items():
                if 'gating' in method_name:
                    print(f'\n\nTraining Gating Unit for {method_name} on class {class_name}...')
                    method.run_late_fusion()
                else:
                    pass            

    


    def evaluate(self, class_name, output_dir):
        metrics_data = []

        test_loader = get_data_loader("test", class_name=class_name, img_size=self.image_size, args=self.args)

        # Paths for predict maps
        rgb_paths = []
        infra_paths = []
        pc_paths = []

        with torch.no_grad():
            for sample, label, pixel_mask, paths in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
                for method in self.methods.values():
                    method.predict(sample, label, pixel_mask)

                    rgb_paths.append(paths[0])
                    infra_paths.append(paths[1])
                    pc_paths.append(paths[2])

                    torch.cuda.empty_cache()

        for method_name, method in self.methods.items():
            method.calculate_metrics()
            metrics = {
                "Method": method_name,
                "Image_ROCAUC": round(method.image_rocauc, 3),
                "RGB_Pixel_ROCAUC": round(method.rgb_pixel_rocauc, 3),
                "RGB_Pixel_F1": round(method.rgb_pixel_f1, 3),
                "RGB_Pixel_AUPR": round(method.rgb_pixel_aupr, 3),
                "RGB_Pixel_AP": round(method.rgb_pixel_ap, 3),
                "Infra_Pixel_ROCAUC": round(method.infra_pixel_rocauc, 3),
                "Infra_Pixel_F1": round(method.infra_pixel_f1, 3),
                "Infra_Pixel_AUPR": round(method.infra_pixel_aupr, 3),
                "Infra_Pixel_AP": round(method.infra_pixel_ap, 3),
                "PC_Pixel_ROCAUC": round(method.pc_pixel_rocauc, 3),
                "PC_Pixel_F1": round(method.pc_pixel_f1, 3),
                "PC_Pixel_AUPR": round(method.pc_pixel_aupr, 3),
            }
 
            metrics_data.append(metrics)

            print(f'Method:{method_name}, Class: {class_name}, Image ROCAUC: {method.image_rocauc:.3f}')
            print(f'Method:{method_name}, Class: {class_name}, RGB Pixel ROCAUC: {method.rgb_pixel_rocauc:.3f}, F1: {method.rgb_pixel_f1:.3f}, AUPR: {method.rgb_pixel_aupr:.3f}, AP: {method.rgb_pixel_ap:.3f}')
            print(f'Method:{method_name}, Class: {class_name}, Infra Pixel ROCAUC: {method.infra_pixel_rocauc:.3f}, F1: {method.infra_pixel_f1:.3f}, AUPR: {method.infra_pixel_aupr:.3f}, AP: {method.infra_pixel_ap:.3f}')
            print(f'Method:{method_name}, Class: {class_name}, PC Pixel ROCAUC: {method.pc_pixel_rocauc:.3f}, F1: {method.pc_pixel_f1:.3f}, AUPR: {method.pc_pixel_aupr:.3f}')

        metrics_df = pd.DataFrame(metrics_data)

        # prediction maps
        # if "RGB" in self.args.method_name:
        #     method.save_prediction_maps(output_dir, rgb_paths, infra_paths, pc_paths, save_num=10, mode="rgb")
        # if "PC" in self.args.method_name:
        #     method.save_prediction_maps(output_dir, rgb_paths, infra_paths, pc_paths, save_num=10, mode="xyz")
        # if "Infra" in self.args.method_name:
        #     method.save_prediction_maps(output_dir, rgb_paths, infra_paths, pc_paths, save_num=10, mode="infra")


        return metrics_df