
import argparse
from runner import Tester
from dataset import mulsen_classes
import pandas as pd
import timm
import torch

def run_3d_ads(args):

    classes = mulsen_classes()

    METHOD_NAMES = [args.method_name]

    image_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    
    for cls in classes:
        model = Tester(args)
        model.fit(cls)
     
        torch.cuda.empty_cache()
        image_rocaucs = model.evaluate(cls)
   
        image_rocaucs_df[cls.title()] = image_rocaucs_df['Method'].map(image_rocaucs)
 
        torch.cuda.empty_cache()
        print(f"\nFinished running on class {cls}")
        print("################################################################################\n\n")

    image_rocaucs_df['Mean'] = round(image_rocaucs_df.iloc[:, 1:].mean(axis=1),3)


    print("\n\n################################################################################")
    print("############################# Image ROCAUC Results #############################")
    print("################################################################################\n")
    print(image_rocaucs_df.to_markdown(index=False))



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
    parser.add_argument('--total_epochs', default=200,
                        help='Training epoches.')
    parser.add_argument('--lr', default=1e-3,
                        help='Save predicts results.')
    parser.add_argument('--weight_decay', default=1e-2,
                        help='Save predicts results.')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')


    args = parser.parse_args()
    run_3d_ads(args)
