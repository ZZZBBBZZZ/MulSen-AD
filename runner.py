import torch
from tqdm import tqdm
import os
from utils_loc import misc
from feature_extractors import multiple_features
        
from dataset import get_data_loader
from models.models import Model, Mlp
import time
class Tester():
    def __init__(self, args):
        self.args = args
        self.image_size = args.img_size
        self.count = 10
        if args.method_name == 'RGB':
            self.methods = {
                "RGB": multiple_features.RGBFeatures(args),
            }
        elif args.method_name == 'Infra':
            self.methods = {
                "Infra": multiple_features.InfraFeatures(args),
            }
        elif args.method_name == 'PC':
            self.methods = {
                "PC": multiple_features.PCFeatures(args),
            }
        elif args.method_name == 'PC+RGB+add':
            self.methods = {
                "PC+RGB+add": multiple_features.PCRGBAddFeatures(args),
            }
        elif args.method_name == 'PC+Infra+add':
            self.methods = {
                "PC+Infra+add": multiple_features.PCInfraAddFeatures(args),
            }
        elif args.method_name == 'RGB+Infra+add':
            self.methods = {
                "RGB+Infra+add": multiple_features.RGBInfraAddFeatures(args),
            }
        elif args.method_name == 'PC+RGB+Infra+add':
            self.methods = {
                "PC+RGB+Infra+add": multiple_features.TripleAddFeatures(args),
            }
        elif args.method_name == 'PC+RGB+gating':
            self.methods = {
                "PC+RGB+gating": multiple_features.PCRGBGatingFeatures(args),
            }
        elif args.method_name == 'PC+Infra+gating':
            self.methods = {
                "PC+Infra+gating": multiple_features.PCInfraGatingFeatures(args),
            }
        elif args.method_name == 'RGB+Infra+gating':
            self.methods = {
                "RGB+Infra+gating": multiple_features.RGBInfraGatingFeatures(args),
            }
        elif args.method_name == 'PC+RGB+Infra+gating':
            self.methods = {
                "PC+RGB+Infra+gating": multiple_features.TripleRGBInfraPointFeatures(args),
            }


    def fit(self, class_name):
        train_loader = get_data_loader("train", class_name=class_name, img_size=self.image_size, args=self.args)

        flag = 0

        print(f'Extracting train features for class {class_name}')
        for sample, _ in train_loader:
            for method in self.methods.values():
                if self.args.save_feature:
                    method.add_sample_to_mem_bank(sample, class_name=class_name)
                else:
                    method.add_sample_to_mem_bank(sample)
                flag += 1
            if flag > self.count:
                flag = 0
                break

        for method_name, method in self.methods.items():
            print(f'\n\nRunning coreset for {method_name} on class {class_name}...')
            method.run_coreset()
            
        if self.args.memory_bank == 'multiple':    
            flag = 0
            print(f'MultiScoring for {method_name} on class {class_name}..')
            for sample, _ in train_loader: 
                for method_name, method in self.methods.items():
                    method.add_sample_to_late_fusion_mem_bank(sample)
                    flag += 1
                if flag > self.count:
                    flag = 0
                    break

            for method_name, method in self.methods.items():
                if 'gating' in method_name:
                    print(f'\n\nTraining Gating Unit for {method_name} on class {class_name}...')
                    method.run_late_fusion()
                else:
                    pass            

    def evaluate(self, class_name):
        image_rocaucs = dict()

        test_loader = get_data_loader("test", class_name=class_name, img_size=self.image_size, args=self.args)
        path_list = []
        with torch.no_grad():     
            for sample, label, rgb_path in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
                for method in self.methods.values():
                    method.predict(sample, label)
                    path_list.append(rgb_path)
                    torch.cuda.empty_cache()
                        

        for method_name, method in self.methods.items():
            method.calculate_metrics()
            image_rocaucs[method_name] = round(method.image_rocauc, 3)

            print(
                f'Class: {class_name}, {method_name} Image ROCAUC: {method.image_rocauc:.3f}, {method_name}')
      
        return image_rocaucs
   
