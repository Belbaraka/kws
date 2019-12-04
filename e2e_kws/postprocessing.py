import os, sys, re
import numpy as np
import json
from python_speech_features import mfcc
from tqdm import tqdm
import scipy.io.wavfile as wav

def generate_windows(signal, fs, window_dur=1.0, shift=0.3):
    """
    TODO : add description

    Args:
    signal: 
    fs:
    window_dur:
    shift:
    
    Returns:
    List, paths to json files.
    """ 
    windows = []
    window_size = int(window_dur * fs)
    current_index = 0 
    end_sentence = False
    signal_duration = len(signal)

    while not end_sentence:
        if current_index + window_size < signal_duration:
            windows.append(signal[current_index:current_index + window_size])
            current_index += int(shift * fs)
        else:
            windows.append(signal[-window_size:])
            end_sentence = True
    return windows



def compute_mfcc_frames(signal, fs, xdim, shift=0.3, num_features=40):
    """
    TODO : add description

    Args:
    signal: 
    fs:
    num_features:
    
    Returns:
    List, paths to json files.
    """     
    windows = generate_windows(signal, fs=fs, shift=shift)
    frames = []
    
    for window in tqdm(windows):
        frame = mfcc(window, samplerate=16000, winlen=0.030, winstep=0.01, numcep=40, 
                     lowfreq=20, highfreq=4000, appendEnergy=False, nfilt=40)
        frames.append(frame)
    return np.array(frames).reshape(-1, xdim, num_features, 1)



def create_file(path2dataset='/aimlx/Datasets/TEDLIUM_release1/'):
    """
    TODO : add description

    Args:
    path2dataset
    
    Returns:
    List, paths to json files.
    """      
    file_partition = []
    
    current_path = os.path.join(path2dataset, 'dev', 'wav')
    for _, _, files in os.walk(current_path):
        for file in files:
            file_partition.append(('dev',  os.path.join(current_path,file)))

    current_path = os.path.join(path2dataset, 'test', 'wav')
    for _, _, files in os.walk(current_path):
        for file in files:
            file_partition.append(('test',  os.path.join(current_path,file)))

    current_path = os.path.join(path2dataset, 'train', 'wav')
    for _, _, files in os.walk(current_path):
        for file in files:
            file_partition.append(('train', os.path.join(current_path,file)))                

    #pickle.dump(file_partition, fp)
    return file_partition



def produce_groundTruth_labels(sentence, label_kw, label_non_kw, start_kw, end_kw, fs=16000, window_dur=1.0, shift=0.1, percentage_kw=0.8):
    """
    TODO : add description

    Args:
    sentence:
    label_kw:
    label_non_kw:
    start_kw:
    end_kw:
    fs:
    window_dur:
    shift:
    percentage_kw:
    
    Returns:
    List, paths to json files.
    """      
    
    gt_labels = []    
    window_size = int(window_dur * fs)
    current_index = 0 
    end_sentence = False
    signal_duration = len(sentence)
    nb_samples_kw = int(end_kw * fs) - int(start_kw * fs) 
    
    while not end_sentence:
        if current_index + window_size < signal_duration:
            if (current_index + window_size < start_kw * fs) or (current_index > end_kw * fs):
                gt_labels.append(label_non_kw)
            else:
                beg_kw_window = max(current_index, int(start_kw * fs))
                end_kw_window = min(current_index + window_size, int(end_kw * fs))
                kw_samples_in_window = end_kw_window - beg_kw_window
                
                if kw_samples_in_window / nb_samples_kw < percentage_kw:
                    gt_labels.append(label_non_kw)
                else:
                    gt_labels.append(label_kw)
            current_index += int(shift * fs)
        else:
            if int(end_kw * fs) < signal_duration - window_size:
                gt_labels.append(label_non_kw)
            elif int(start_kw * fs) > signal_duration - window_size:
                gt_labels.append(label_kw)
            else:
                kw_samples_in_window = int(end_kw * fs) - (signal_duration - window_size) 
                
                if kw_samples_in_window / nb_samples_kw < percentage_kw:
                    gt_labels.append(label_non_kw)
                else:
                    gt_labels.append(label_kw)                
                
            end_sentence = True
            
    return np.array(gt_labels)



def extract_sentence(path2wav_file, path2dataset, file_partition, keywords, duration=5, shift=0.1, percentage_kw=1.0):
    """
    TODO : add description

    Args:
    path2wav_file:
    path2dataset:
    file_partition:
    keywords:
    duration:
    shift:
    percentage_kw:
    
    Returns:
    fs:
    sentence:
    y_test:
    """      
    
    filename = path2wav_file.split('/')[-1]
    occurence = int(filename.split('_')[-1].split('*')[0])
    filename = '_'.join(filename.split('_')[:2])
    keyword = path2wav_file.split('/')[-2]
    
    filtered_list = list(filter(lambda x: filename in x[1], file_partition))[0]
    partition = filtered_list[0]
    path2wav_talk = filtered_list[1]
        
    path2json = os.path.join(path2dataset, partition, 'final_json', filename + '.json')
    with open(path2json) as json_file:
        data = json.load(json_file)
        count = -1 
        for word in data['words']:
            if partition == 'train':
                word = data['words'][word]
            if not word['case'] == 'not-found-in-audio' and (word['alignedWord'] == keyword):
                count += 1
            if count == occurence:
                start_kw, end_kw = word['start'], word['end']
                break

    if start_kw - duration/2.0 < 0:
        start_sentence = 0
        end_sentence = duration - end_kw
    else:
        start_sentence = start_kw - duration/2.0
        end_sentence = end_kw + duration/2.0
        
    fs, signal = wav.read(path2wav_talk)
    start_signal, end_signal = int(start_sentence * fs), int(end_sentence * fs)
    sentence = signal[start_signal:end_signal]
    
    y_test = produce_groundTruth_labels(sentence, label_kw=keywords.index(keyword), label_non_kw=len(keywords), 
                                        start_kw=start_kw-start_sentence, end_kw=start_kw-start_sentence + (end_kw -start_kw), 
                                        fs=fs, shift=shift, percentage_kw=percentage_kw)
    
    return fs, sentence, y_test



def generate_windows_indexes(signal_duration, fs, window_dur=1.0, shift=0.3):
    """
    TODO : add description

    Args:
    signal_duration:
    fs:
    window_dur:
    shift:
    
    Returns:
    windows_indexes:
    """       
    windows_indexes = []
    window_size = int(window_dur * fs)
    current_index = 0 
    end_sentence = False

    while not end_sentence:
        if current_index + window_size < signal_duration:
            windows_indexes.append((current_index,current_index + window_size))
            current_index += int(shift * fs)
        else:
            windows_indexes.append((signal_duration - window_size, signal_duration))
            end_sentence = True
    return windows_indexes



def get_offset(path2transcription):
    """
    Get the moment in the audio file where the TED speaker starts talking

    Args:
    path2transcription: File path of the transcription (.stm file).

    Returns:
    Float, the offset value.
    """
    with open(path2transcription, 'rt') as f:
        records = f.readlines()
        for sentence in records:
            fields = sentence.split()
            label = fields[6:]
            if not 'ignore_time_segment_in_scoring' in label:
                return float(fields[3])
    return



def extract_keyword_occurences(path2json_TED_talk, keyword, fs, is_from_train=False):
    """
    TODO : add description

    Args:
    path2json_TED_talk:
    keyword:
    fs:
    is_from_train:
    
    Returns:
    keyword_occurences:
    
    """       
    keyword_occurences = []
    
    with open(path2json_TED_talk) as json_file:
        data = json.load(json_file)
        for word in data['words']:
            if is_from_train:
                word = data['words'][word]
                if not word['case'] == 'not-found-in-audio' and (word['alignedWord'] == keyword):
                    start_kw, end_kw = word['start'], word['end']
                    keyword_occurences.append((int(start_kw * fs), int(end_kw * fs)))
            else: 
                path2stm_file = path2json_TED_talk.replace('final_json', 'stm').replace('.json', '.stm')
                offset = get_offset(path2transcription=path2stm_file)
                if not word['case'] == 'not-found-in-audio' and (word['alignedWord'] == keyword):
                    start_kw, end_kw = word['start'], word['end']
                    keyword_occurences.append((int((offset + start_kw) * fs), int((offset + end_kw) * fs)))
    return keyword_occurences



def pre_gt_vector(windows_indexes, keyword_occurences, non_kw_label, kw_label):
    """
    TODO : add description

    Args:
    windows_indexes:
    keyword_occurences:
    non_kw_label:
    kw_label:
    
    Returns:
    kw_gt_vector:
    
    """       
    kw_gt_vector = [non_kw_label] * len(windows_indexes)
    
    for occurence in keyword_occurences:
        start_kw, end_kw = occurence[0], occurence[1]
        windows_kw = list(filter(lambda x: start_kw > x[0] and end_kw < x[1], windows_indexes))
        for w in windows_kw:
            kw_gt_vector[windows_indexes.index(w)] = kw_label
    return kw_gt_vector



def merge_gt_vectors(gt_vector1, gt_vector2, non_kw_label, undefined_label=-1):
    """
    TODO : add description

    Args:
    gt_vector1:
    gt_vector2:
    non_kw_label:
    undefined_label:
    
    Returns:
    merged_vector:
    
    """     
    merged_vector = []
    for i in range(len(gt_vector1)):
        if gt_vector1[i] ==  gt_vector2[i]:
            merged_vector.append(gt_vector1[i])
        else:
            if gt_vector1[i] == non_kw_label:
                merged_vector.append(gt_vector2[i])
            elif gt_vector2[i] == non_kw_label:
                merged_vector.append(gt_vector1[i])
            else:
                merged_vector.append(undefined_label)
    return merged_vector



def extract_gt_vector(windows_indexes, path2json_TED_talk, keywords, fs, is_from_train=False):
    """
    TODO : add description

    Args:
    windows_indexes:
    path2json_TED_talk:
    keywords:
    fs:
    is_from_train:
    
    Returns:
    gt_vector:
    
    """   
    non_keyword_label = len(keywords)
    gt_vector = [non_keyword_label] * len(windows_indexes)
    
    for kw in tqdm(keywords):
        keyword_occurences = extract_keyword_occurences(path2json_TED_talk, kw, fs, is_from_train)
        kw_gt_vector = pre_gt_vector(windows_indexes, keyword_occurences, non_keyword_label, kw_label=keywords.index(kw))
        gt_vector = merge_gt_vectors(gt_vector, kw_gt_vector, non_kw_label=non_keyword_label, undefined_label=-1)
        
    return np.array(gt_vector)