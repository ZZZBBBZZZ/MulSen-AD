"""
Code based on the official M3DM code found at
https://github.com/nomewang/M3DM
and PatchCore found at
https://github.com/amazon-science/patchcore-inspection
"""
"""
# Copyright (c) 2023 nomewang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""

import argparse
from runner import Tester
from dataset import mulsen_classes
import pandas as pd
import timm
import torch
import os
from datetime import datetime
import random
import numpy as np
def run_3d_ads(args):
    if args.random_state is not None:
        set_random_seed(args.random_state)
    classes = mulsen_classes()

    METHOD_NAMES = [args.method_name]


    all_metrics_df = pd.DataFrame()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(args.output_dir, args.method_name, timestamp)
    for cls in classes:
        output_dir = os.path.join(args.output_dir, cls)
      
        model = Tester(args)
        model.fit(cls)
     
        torch.cuda.empty_cache()
  
        metrics_df = model.evaluate(cls, output_dir)
        metrics_df['Class'] = cls.title()
        all_metrics_df = pd.concat([all_metrics_df, metrics_df], ignore_index=True)
 
 
        torch.cuda.empty_cache()
        print(f"\nFinished running on class {cls}")
        print("################################################################################\n\n")

    metric_columns = [col for col in all_metrics_df.columns if col not in ['Method', 'Class']]

    mean_row = all_metrics_df[metric_columns].mean(axis=0, skipna=True)

    mean_row_data = {col: mean_row.get(col, None) for col in metric_columns}
    mean_row_data['Method'] = 'Mean'
    mean_row_data['Class'] = 'Overall'

    all_metrics_df = pd.concat([all_metrics_df, pd.DataFrame([mean_row_data])], ignore_index=True)


    # Output the DataFrame with the mean if needed
    print("\n\n################################################################################")
    print("############################# Metrics with Mean ###############################")
    print("################################################################################\n")
    print(all_metrics_df.to_markdown(index=False))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir,exist_ok=True)
    
    # After all classes, save the entire metrics DataFrame
    output_file = os.path.join(args.output_dir, "all_metrics_results.csv")
    all_metrics_df.to_csv(output_file, index=False)

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--method_name', default='PC+RGB+Infra+gating', type=str, 
                        help='Anomaly detection modal name.')
    parser.add_argument('--memory_bank', default='multiple', type=str,
                        choices=["multiple", "single"],
                        help='memory bank mode: "multiple", "single".')
    parser.add_argument('--rgb_backbone_name', default='vit_base_patch8_224_dino', type=str, 
                        choices=['vit_base_patch8_224_dino', 'vit_base_patch8_224', 'vit_base_patch8_224_in21k', 'vit_small_patch8_224_dino'],
                        help='Timm checkpoints name of RGB backbone.')
    parser.add_argument('--xyz_backbone_name', default='Point_MAE', type=str, choices=['Point_MAE', 'Point_Bert'],
                        help='Checkpoints name of RGB backbone[Point_MAE, Point_Bert].')
    parser.add_argument('--fusion_module_path', default='checkpoints/checkpoint-0.pth', type=str,
                        help='Checkpoints for fusion module.')
    parser.add_argument('--save_feature', default=True, action='store_true',
                        help='Save feature for training fusion block.')
    parser.add_argument('--save_preds', default=False, action='store_true',
                        help='Save predicts results.')
    parser.add_argument('--group_size', default=128, type=int,
                        help='Point group size of Point Transformer.')
    parser.add_argument('--num_group', default=1024, type=int,
                        help='Point groups number of Point Transformer.')
    parser.add_argument('--random_state', default=None, type=int,
                        help='random_state for random project')
    parser.add_argument('--dataset_path', default='./dataset/MulSen_AD', type=str, 
                        help='Dataset store path')
    parser.add_argument('--img_size', default=224, type=int,
                        help='Images size for model')
    parser.add_argument('--coreset_eps', default=0.9, type=float,
                        help='eps for sparse project')
    parser.add_argument('--f_coreset', default=0.1, type=float,
                        help='eps for sparse project')
    parser.add_argument('--asy_memory_bank', default=None, type=int,
                        help='build an asymmetric memory bank for point clouds')
    parser.add_argument('--ocsvm_nu', default=0.5, type=float,
                        help='ocsvm nu')
    parser.add_argument('--ocsvm_maxiter', default=1000, type=int,
                        help='ocsvm maxiter')
    parser.add_argument('--rm_zero_for_project', default=False, action='store_true',
                        help='Save predicts results.')
    parser.add_argument('--total_epochs', default=1,
                        help='Training epoches.')
    parser.add_argument('--lr', default=1e-3,
                        help='Save predicts results.')
    parser.add_argument('--weight_decay', default=1e-2,
                        help='Save predicts results.')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')


    args = parser.parse_args()
    run_3d_ads(args)
