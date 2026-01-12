import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb/Code')
sys.path.append('/home/jiayu/SeismoApnea4Ubicomp_Feb')
import numpy as np
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


# def get_label_binary_Nor_HypApn(events_window):
#     """
#     2 class classification
#     0: Normal 
#     1: Hypopnea + Apnea
#     """
#     unique_events = np.unique(events_window)
    
#     if len(unique_events) == 1:
#         if unique_events[0] == 0: return 0
#         else: return 1

#     if len(unique_events) == 2 and 0 in unique_events:
#         non_zero_event = unique_events[unique_events != 0][0]
#         non_zero_positions = np.where(events_window == non_zero_event)[0]
#         if len(non_zero_positions) >= 100: return 1
#         else: return None

#     return None


def get_label_binary_Nor_HypApn(events_window):
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
        else: return 0

    return 0



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



id2fold = {4: '1', 9: '1', 23: '1', 26: '1', 30: '1', 35: '1', 39: '1', 48: '1', 50: '1', 53: '1', 56: '1', 59: '1', 63: '1', 84: '1', 90: '1', 93: '1', 99: '1', 102: '1', 108: '1', 114: '1', 118: '1', 136: '1', 138: '1', 143: '1', 144: '1', 150: '1', 151: '1', 155: '1',
              12: '2', 16: '2', 18: '2', 25: '2', 28: '2', 29: '2', 31: '2', 36: '2', 41: '2', 45: '2', 51: '2', 58: '2', 60: '2', 66: '2', 68: '2', 71: '2', 97: '2', 101: '2', 104: '2', 105: '2', 107: '2', 110: '2', 111: '2', 116: '2', 127: '2', 131: '2', 154: '2',
              1: '3', 10: '3', 11: '3', 15: '3', 22: '3', 32: '3', 33: '3', 34: '3', 38: '3', 42: '3', 44: '3', 52: '3', 64: '3', 87: '3', 91: '3', 92: '3', 96: '3', 98: '3',106: '3',119: '3',121: '3',123: '3',130: '3',141: '3',153: '3',156: '3',157: '3',
              2: '4', 3: '4', 5: '4, ',7: '4',8: '4',14: '4',17: '4',19: '4',24: '4',27: '4',40: '4',46: '4',55: '4',57: '4',69: '4',81: '4',83: '4',86:'4',95:'4' ,117:'4' ,120:'4' ,122:'4' ,134:'4' ,137:'4' ,146:'4' ,148:'4' ,152:'4'}

def analyze_and_save_fold_distributions():
    label_func_map = {
    'binary_Nor_HypApn': get_label_binary_Nor_HypApn,
    'binary_Sleep_Wake': get_label_binary_Sleep_Wake,
    }
    
    TIME_INDEX = -3
    data_length = 90

    data_dir = f'Data/data_{data_length}s_30s_rdi/'
    infos = read_excel()

    patient_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    # patient_indices = np.arange(len(patient_files))

    save_dir = f'Data/fold_data_p{109}_MTL_{data_length}s/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)


    for fold in range(1, 5):
        fold_path = os.path.join(save_dir, f'fold{fold}.npy')

        
        all_processed_data = []  
        patient_lengths = []  
        patient_cnt = 0
        print("Loading and processing patient data...")
        for patient_file in patient_files:
            print(f'\nLoading {patient_file}...')
            patient_data = np.load(os.path.join(data_dir, patient_file))
            if len(patient_data) == 0: 
                print(f'Skipping empty data for {patient_file}.')
                continue
            patient_id = np.unique(patient_data[:, -2])[0]
            print(f'Patient ID: {patient_id}')

            if id2fold.get(int(patient_id)) != str(fold):
                print(f'Skipping patient {patient_id} for fold {fold}. Assigned to fold {id2fold.get(int(patient_id))}.')
                continue


            print(f'Before Sleep Start Time filtering: {patient_data.shape}')

            # Sleep start time and end time filtering
            sleep_start_unix, sleep_over_unix = cal_sleep_startime(infos, patient_file)
            times = patient_data[:, TIME_INDEX]
            mask_time = (times >= sleep_start_unix) & (times <= sleep_over_unix)
            patient_data = patient_data[mask_time]


            
        #     # Split
            events_windows = patient_data[:, -2*data_length*10-4:-data_length*10-4]
            # stages_windows = patient_data[:, 30060: 30120]
            stages_windows = patient_data[:, -3*data_length*10-4:-2*data_length*10-4]


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
                
                if label_apn is not None:
                    valid_indices.append(idx)
                    patient_labels_sleep.append(label_sleep)
                    patient_labels_apn.append(label_apn)

            print(f"Total samples: {len(patient_labels_sleep)}")

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

        fold_data = all_processed_data
  
        np.save(fold_path, fold_data)
            

        fold_labels_apn = fold_data[:, -1]
        print(f"\nFold {fold} label distribution:")
        fold_label_dist = np.unique(fold_labels_apn, return_counts=True)
        for label, count in zip(fold_label_dist[0], fold_label_dist[1]):
            print(f"Label {label}: {count} samples ({count/len(fold_labels_apn)*100:.2f}%)")


if __name__ == "__main__":
    analyze_and_save_fold_distributions()