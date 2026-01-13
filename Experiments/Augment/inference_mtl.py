import torch
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
from Code.utils import npy2dataset_inference_true_MTL
from Code.models.clf import ApneaClassifier_PatchTST_MTL
from Code.utils_dl import inference, find_best_threshold_F1, inference_MTL, find_best_threshold_F1_and_Bacc
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, roc_curve
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.metrics import confusion_matrix
import seaborn as sns
import yaml


plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_checkpoint(model_folder):
    sorted_files = sorted(os.listdir(model_folder), key=lambda x: int(x.split('_')[0][5:]), reverse=False)
    print(f'sorted_files: {sorted(os.listdir(model_folder))}')

    model_path = model_folder + sorted_files[-1]
    print(f'model_path: {model_path}')
    
    checkpoint = torch.load(model_path, weights_only=False)
    print(checkpoint.keys())

    val_bacc_stage = checkpoint['val_bacc_stage']
    val_f1_stage = checkpoint['val_f1_stage']
    val_bacc_apnea = checkpoint['val_bacc_apnea']
    val_f1_apnea = checkpoint['val_f1_apnea']
    print(f'val_bacc_stage: {val_bacc_stage}, val_f1_stage: {val_f1_stage}, val_bacc_apnea: {val_bacc_apnea}, val_f1_apnea: {val_f1_apnea}')

    return checkpoint


def threshold_adjustment(val_probs_pos, labels, legend):
    threshold_results = find_best_threshold_F1(
        y_true=np.array(labels),
        y_prob=val_probs_pos,
        num_thresholds=500,
        average='weighted'
    )
    best_threshold = threshold_results['best_threshold']
    bacc = threshold_results['best_bacc']
    f1 = threshold_results['best_f1']
    cm = threshold_results['confusion_matrix']
    # preds = threshold_results['y_pred']


    print(f'Best Threshold ({legend}): {best_threshold}')
    print(f'Balanced Accuracy at Best F1 Threshold ({legend}): {bacc}')
    print(f'Weighted F1 at Best F1 Threshold ({legend}): {f1}')
    print(f'Confusion Matrix at Best F1 Threshold ({legend}):\n{cm}')

    return best_threshold



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apnea Detection - BSG')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sleep_index', type=int, default=30060)
    parser.add_argument('--nseg', type=int, default=60)
    parser.add_argument('--seq_len', type=int, default=590)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2) 
    parser.add_argument('--XYZ', type=str, default='XY')
    parser.add_argument('--raw', type=str, default=False)
    parser.add_argument('--tta_method', type=str, default='avgnew')
    # parser.add_argument('--tta_method', type=str, default='avg')
    parser.add_argument('--supcon', type=bool, default=True)
    parser.add_argument('--gama', type=float, default=2e-5)
    parser.add_argument('--threhold', type=str, default='0.5')
    parser.add_argument('--Index', type=str, default='AHI')
    parser.add_argument('--write_yaml', type=bool, default=True)
    parser.add_argument('--std', type=float, default=7)
    args = parser.parse_args()

    Type = 'MTL'
    Model = 'PatchTST' 


    ID_num = 109
    model_folder_name = f'{args.XYZ}_60s_F1_ws1_wa1_std{args.std}'


    fold_id_mapping = {}
    fold_id_threshold_stage = {}
    fold_id_threshold_apnea = {}

    for fold_idx in range(1, 5):
        fold_idx = int(fold_idx)

        data_path = f'Data/fold_data_p{ID_num}_{Type}_60s/'
        yaml_path = f'Experiments/Augment/configs/{model_folder_name}.yaml'

        val_loader, test_loader = npy2dataset_inference_true_MTL(data_path, fold_idx, args)
        cuda = '1'
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        

        if Model == 'PatchTST':
            patch_len = 24
            n_layers = 4
            d_model = 64
            n_heads = 4
            d_ff = 256
            output_class = 2
            mask_ratio = 0.5

            model = ApneaClassifier_PatchTST_MTL(
                input_size=2, num_classes=output_class,
                seq_len=args.seq_len, patch_len=patch_len,
                stride=patch_len // 2,
                n_layers=n_layers, d_model=d_model,
                n_heads=n_heads, d_ff=d_ff,
                axis=3 if args.XYZ == 'XYZ' else 2,
                dropout=args.dropout,
                mask_ratio=mask_ratio,
                ).to(device)
            
            print(f'fold_idx: {fold_idx}')
            model_folder = f'Experiments/Augment/Models/{model_folder_name}/fold{fold_idx}/PatchTST_patchlen{patch_len}_nlayer{n_layers}_dmodel{d_model}_nhead{n_heads}_dff{d_ff}/'

            print(f'model_folder: {model_folder}')

        checkpoint = load_checkpoint(model_folder)
        model.load_state_dict(checkpoint['model_state_dict'])

        preds_stage, labels_stage, probs_stage, preds_apnea, labels_apnea, probs_apnea, others, stage_apnea_mask = inference_MTL(model, test_loader, device, tta_method=args.tta_method, std=args.std)


        best_threshold_stage = threshold_adjustment(np.array(probs_stage)[:, 1], labels_stage, 'Stage')
        best_threshold_apnea = threshold_adjustment(np.array(probs_apnea)[:, 1], labels_apnea, 'Apnea')


        fold_id_threshold_stage[str(fold_idx)] = float(best_threshold_stage)
        fold_id_threshold_apnea[str(fold_idx)] = float(best_threshold_apnea)
        fold_id_mapping[str(fold_idx)] = np.unique(others[:, -2]).tolist()
        print(f'fold_id_mapping[{str(fold_idx)}]: {fold_id_mapping[str(fold_idx)]}')
    
    



        if args.write_yaml:
            with open(yaml_path, 'w') as yaml_file:
                yaml.dump({'fold_id_mapping': fold_id_mapping, 'fold_to_threshold_stage': fold_id_threshold_stage, 'fold_to_threshold_apnea': fold_id_threshold_apnea}, yaml_file)


        print()
        print()

