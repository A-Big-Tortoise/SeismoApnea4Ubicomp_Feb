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

    # val_bacc_stage = checkpoint['val_bacc_stage']
    # val_f1_stage = checkpoint['val_f1_stage']
    # val_bacc_apnea = checkpoint['val_bacc_apnea']
    # val_f1_apnea = checkpoint['val_f1_apnea']
    val_cm = checkpoint['val_cm']
    # print(f'val_bacc_stage: {val_bacc_stage}, val_f1_stage: {val_f1_stage}, val_bacc_apnea: {val_bacc_apnea}, val_f1_apnea: {val_f1_apnea}')
    print(f'val_cm:\n{val_cm}')
    return checkpoint



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


    fold_id_mapping = {}
    fold_id_threshold_stage = {}
    fold_id_threshold_apnea = {}

    for fold_idx in range(1, 6):
        fold_idx = int(fold_idx)


        cuda = '1'
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        


        print(f'fold_idx: {fold_idx}')
        model_folder = f'/home/jiayu/SeismoApnea4Ubicomp_Feb/Experiments/Yingjian/Models/Y_Yingjian_60s_times10/fold{fold_idx}/PatchTST_patchlen24_nlayer5_dmodel64_nhead4_dff256/'
        print(f'model_folder: {model_folder}')

        checkpoint = load_checkpoint(model_folder)


