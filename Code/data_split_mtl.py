import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from Code.utils import read_excel, time2timestamp


def get_patient_info_from_filename(file_name):
    patient_name = file_name.split('_')[0]
    patient_time = (file_name.split('_')[1].split('.')[0] + '.' + file_name.split('_')[1].split('.')[1])
    patient_id = file_name.split('.')[0] + '.' + file_name.split('.')[1]
    return patient_name, patient_time, patient_id
    

def get_label_binary_Sleep_Wake(events_window):
    """
    2 class classification
    0: Sleep
    1: Wake
    """ 
    ns_num = np.sum(events_window == -1)
    sleep_num = np.sum((events_window >= 0) & (events_window <= 3))
    wake_num = np.sum(events_window == 4)

    if ns_num > 0 or wake_num > 0: 
        return 1
    else:
        return 0


def get_label_binary_Nor_HypApn_2(events_window):
    """
    2 class classification
    0: Normal 
    1: Hypopnea + Apnea
    """
    unique_events = np.unique(events_window)
    
    if len(unique_events) == 1:
        if unique_events[0] == 0: return 0
        else: return 1

    if len(unique_events) == 2 and 0 in unique_events:
        non_zero_event = unique_events[unique_events != 0][0]
        non_zero_positions = np.where(events_window == non_zero_event)[0]
        if len(non_zero_positions) >= 100: return 1
        else: return -1

    return -1




def cal_sleep_startime(infos, patient_file):
    patient_name_fn, patient_time_fn, patient_id_fn = get_patient_info_from_filename(patient_file) 
    print(f'patient info from filename: {patient_name_fn}, {patient_time_fn}, {patient_id_fn}')
    # find the corresponding patient info in excel file

    info = None
    for info_temp in infos:
        if patient_name_fn in info_temp[0] and info_temp[-1] == patient_time_fn:
            info = info_temp
            break
   
    # Calculate sleep start time and convert to unix timestamp
    try: lights_out, sleep_latency, lights_on = info[2], info[3], info[4]
    except TypeError: print(info)

    sleep_start = lights_out + pd.Timedelta(minutes=sleep_latency) + pd.Timedelta(minutes=5)
    sleep_start_unix = time2timestamp(sleep_start)
    sleep_over = lights_on - pd.Timedelta(minutes=3)
    sleep_over_unix = time2timestamp(sleep_over)

    print(f"Info from excel: {info}")
    print(f"Sleep start time: {sleep_start}")
    print(f"Sleep over time: {sleep_over}")

    return sleep_start_unix, sleep_over_unix



def analyze_and_save_fold_distributions():
    # dataset [x, y, z, tho, abd, Body, events, mac, time, ID, Room]

    label_func_map = {
    'binary_Nor_HypApn': get_label_binary_Nor_HypApn_2,
    'binary_Sleep_Wake': get_label_binary_Sleep_Wake,
    }
    
    TIME_INDEX = -3
    data_length = 60


    data_dir = f'/home/jiayu/SleepApnea4Ubicomp/Data/data_p144_rdi_45s/'
    infos = read_excel()

    patient_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    # patient_indices = np.arange(len(patient_files))

    save_dir = f'/home/jiayu/SleepApnea4Ubicomp/Data/fold_data_p{len(patient_files)}_MTL_45s_minus1/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    all_processed_data = []  
    patient_lengths = []  
    patient_cnt = 0
    print("Loading and processing patient data...")
    for patient_file in patient_files:
        print(f'\nLoading {patient_file}...')
        patient_data = np.load(os.path.join(data_dir, patient_file))

        print(f'Before Sleep Start Time filtering: {patient_data.shape}')

        if len(patient_data) <= 50:
            print("No data found for this patient. Skipping...")
            continue

        # Sleep start time and end time filtering
        sleep_start_unix, sleep_over_unix = cal_sleep_startime(infos, patient_file)
        times = patient_data[:, TIME_INDEX]
        mask_time = (times >= sleep_start_unix) & (times <= sleep_over_unix)
        patient_data = patient_data[mask_time]


        
        # Split
        events_windows = patient_data[:, -2*data_length*10-4:-data_length*10-4]
        stages_windows = patient_data[:, 30060: 30120]


        print(f'index of events in data: {patient_data.shape[1]-data_length*10-4} to {patient_data.shape[1]-4}')
        print(f'After Sleep Start Time filtering: {patient_data.shape}')
        print(f"Events shape: {events_windows.shape}")


        valid_indices = []
        patient_labels_apn = []
        patient_labels_sleep = []
        

        for idx in range(len(events_windows)):
            event_window = events_windows[idx]
            stage_window = stages_windows[idx]
            label_sleep = label_func_map['binary_Sleep_Wake'](stage_window)
            label_apn = label_func_map['binary_Nor_HypApn'](event_window)

            valid_indices.append(idx)
            patient_labels_sleep.append(label_sleep)
            patient_labels_apn.append(label_apn)


        if len(valid_indices) > 0: 
            patient_lengths.append(len(valid_indices))

            patient_cnt += 1
            valid_patient_data = patient_data[valid_indices]
            valid_patient_labels_sleep = np.array(patient_labels_sleep).reshape(-1, 1)
            valid_patient_labels_apn = np.array(patient_labels_apn).reshape(-1, 1) 
            processed_patient_data = np.hstack((valid_patient_data, valid_patient_labels_sleep, valid_patient_labels_apn))
            all_processed_data.append(processed_patient_data)
            
            print(f"Valid samples: {len(valid_indices)}")
            print(f"Processed data shape: {processed_patient_data.shape}")
        print('-'*50)

    all_processed_data = np.concatenate(all_processed_data, axis=0)
    print("\nFinal data shape:", all_processed_data.shape)

    # Create cumulative sums for proper indexing
    cumsum_lengths = np.cumsum([0] + patient_lengths)
    patient_indices = np.arange(patient_cnt)
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    for fold_idx, (other_idx, fold_idx_patients) in enumerate(kf.split(patient_indices)):
        print(f"\nProcessing Fold {fold_idx + 1}:")
        print(f"Fold {fold_idx + 1} patients: {[patient_files[i] for i in fold_idx_patients]}")
        
        fold_data = []
        for idx in fold_idx_patients:
            start_idx = cumsum_lengths[idx]
            end_idx = cumsum_lengths[idx + 1]
            fold_data.append(all_processed_data[start_idx:end_idx])
        
        fold_data = np.concatenate(fold_data, axis=0)
        
        fold_dir = save_dir
        # os.makedirs(fold_dir, exist_ok=True)
        
        np.save(os.path.join(fold_dir, 'fold%d.npy'%(fold_idx + 1)), fold_data)
        
        print(f"Fold {fold_idx + 1} samples: {len(fold_data)}")
        print(f"Fold {fold_idx + 1} data shape: {fold_data.shape}")
        
        fold_labels_apn = fold_data[:, -1]
        print(f"\nFold {fold_idx + 1} label distribution:")
        fold_label_dist = np.unique(fold_labels_apn, return_counts=True)
        for label, count in zip(fold_label_dist[0], fold_label_dist[1]):
            print(f"Label {label}: {count} samples ({count/len(fold_labels_apn)*100:.2f}%)")
    print(f"Patient Number: {patient_cnt}")


if __name__ == "__main__":
    analyze_and_save_fold_distributions()