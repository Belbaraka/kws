import os
import numpy as np
import math
import json
from python_speech_features import mfcc
from tqdm import tqdm
import scipy.io.wavfile as wav

def generate_windows(signal, fs, window_dur=1.0, shift=0.3):
    """
    Slices up signal into consecutive, overlapping windows of duration `window_dur` and shifted by `shift`.

    Args:
    signal: signal over which overlapping windows will be extracted
    fs: sampling frequency
    window_dur: window/frame duration (in seconds)
    shift: amount (in seconds) by which windows are shifted
    
    Returns:
    List of the windows extracted from the signal.
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


def mfcc_evaluation_set(signal, fs, windows, xdim=98, num_features=40, verbose=1):
    """
    Function specifically made for the evaluation set.
    Compute MFCC features over overlapping windows extracted from the signal.

    Args:
    signal: signal over which window will be extracted and features computed
    fs: sampling frequency
    windows: list of window indexes on which we want to compute MFCC frames
    xdim: row dimension of the feature matrix
    num_features: number of MFCCs (column dimension of feature matrix)
    
    Returns:
    Numpy array of shape (number_of_frames_extracted, xdim, num_features, 1),
    where the features extracted from the signal are stacked together.
    """     
    frames = []
    
    for w in (tqdm(windows) if verbose else windows):
        sig_window = signal[w[0]:w[1]]
        frame = mfcc(sig_window, samplerate=16000, winlen=0.030, winstep=0.01, numcep=num_features, 
                     lowfreq=20, highfreq=4000, appendEnergy=False, nfilt=num_features)
        frames.append(frame)
    return np.array(frames).reshape(-1, xdim, num_features, 1)


def compute_mfcc_frames(signal, fs, xdim, w_dur=1.0, shift=0.3, num_features=40, verbose=1):
    """
    Compute MFCC features over overlapping windows extracted from the signal.

    Args:
    signal: signal over which window will be extracted and features computed
    fs: sampling frequency
    xdim: row dimension of the feature matrix
    w_dur: window/frame duration (in seconds)
    shift: amount (in seconds) by which windows are shifted
    num_features: number of MFCCs (column dimension of feature matrix)
    
    Returns:
    Numpy array of shape (number_of_frames_extracted, xdim, num_features, 1),
    where the features extracted from the signal are stacked together.
    """     
    windows = generate_windows(signal, fs=fs, window_dur=w_dur, shift=shift)
    frames = []
    
    for window in (tqdm(windows) if verbose else windows):
        frame = mfcc(window, samplerate=16000, winlen=0.030, winstep=0.01, numcep=num_features, 
                     lowfreq=20, highfreq=4000, appendEnergy=False, nfilt=num_features)
        frames.append(frame)
    return np.array(frames).reshape(-1, xdim, num_features, 1)



def create_file(path2dataset='/aimlx/Datasets/TEDLIUM_release1/'):
    """
    Loads all the paths to every .wav TED talk in the dataset.

    Args:
    path2dataset: path to dataset of TED talks
    
    Returns:
    file_partition: list which contains the path to all .wav TED talks.
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

    return file_partition



def produce_groundTruth_labels(sentence, kw_label, non_kw_label, start_kw, end_kw, fs=16000, window_dur=1.0, shift=0.1, percentage_kw=0.8):
    """
    Given a short signal (aka a sentence) extract overlapping windows and assign a label to each one of them (`kw_label` or `non_kw_label`). 
    The sentence should only contain one occurence of the keyword associated to `kw_label` and non keywords.

    Args:
    sentence: short signal which contain a unique occurence of a keyword
    kw_label: label associated to the keyword in sentence
    non_kw_label: non keyword label
    start_kw: start of the keyword in sentence (in seconds)
    end_kw: end of the keyword in sentence (in seconds)
    fs: sampling frequency (Hz)
    window_dur: window/frame duration (in seconds)
    shift: amount (in seconds) by which windows are shifted
    percentage_kw: threshold percentage of keyword to be present in the window/frame to assign it the keyword label.  
    
    Returns:
    Numpy array of labels associated to the windows extracted from sentence.
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
                gt_labels.append(non_kw_label)
            else:
                beg_kw_window = max(current_index, int(start_kw * fs))
                end_kw_window = min(current_index + window_size, int(end_kw * fs))
                kw_samples_in_window = end_kw_window - beg_kw_window
                
                if kw_samples_in_window / nb_samples_kw < percentage_kw:
                    gt_labels.append(non_kw_label)
                else:
                    gt_labels.append(kw_label)
            current_index += int(shift * fs)
        else:
            if int(end_kw * fs) < signal_duration - window_size:
                gt_labels.append(non_kw_label)
            elif int(start_kw * fs) > signal_duration - window_size:
                gt_labels.append(kw_label)
            else:
                kw_samples_in_window = int(end_kw * fs) - (signal_duration - window_size) 
                
                if kw_samples_in_window / nb_samples_kw < percentage_kw:
                    gt_labels.append(non_kw_label)
                else:
                    gt_labels.append(kw_label)                
                
            end_sentence = True
            
    return np.array(gt_labels)



def extract_sentence(path2wav_file, path2dataset, file_partition, keywords, duration=5, shift=0.1, percentage_kw=1.0):
    """
    Extracts a small speech segment around the keyword sample present in the .wav file `path2wav_file`.
    Apply also the function `produce_groundTruth_labels()` on it.

    Args:
    path2wav_file: path the .wav keyword sample
    path2dataset: path to dataset of the TED talks
    file_partition: list which contains the path to all .wav TED talks.
    keywords: list of keywords on which the models were trained on
    duration: duration of the speech segment to be extracted (in seconds)
    shift: amount (in seconds) by which windows are shifted
    percentage_kw: threshold percentage of keyword to be present in the window/frame to assign it the keyword label.  
    
    Returns:
    fs: sampling frequency (Hz)
    sentence: speech segment extracted
    y_test: numpy array of labels associated to the windows extracted from sentence
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
    
    y_test = produce_groundTruth_labels(sentence, kw_label=keywords.index(keyword), non_kw_label=len(keywords), 
                                        start_kw=start_kw-start_sentence, end_kw=start_kw-start_sentence + (end_kw -start_kw), 
                                        fs=fs, shift=shift, percentage_kw=percentage_kw)
    
    return fs, sentence, y_test



def generate_windows_indexes(signal_duration, fs, window_dur=1.0, shift=0.3):
    """
    Get the indices of overlapping windows over a signal of length `signal_duration`
    
    Args:
    signal_duration: length of signal in terms of sample (i.e len(signal))
    fs: sampling frequency (Hz)
    window_dur: window/frame duration (in seconds)
    shift: amount (in seconds) by which windows are shifted
    
    Returns:
    windows_indexes: list of tuples (start, end) of overlapping window indexes
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
    Given a TED talk, extract all the occurences of a given keyword. 
    
    Args:
    path2json_TED_talk: path to the json file which contains the aligned transcript of the TED talk
    keyword: keyword to look for
    fs: sampling frequency (Hz)
    is_from_train: bool, is the TED talk from the train set (json file are constructed slightly differently)
    
    Returns:
    keyword_occurences: list of occurences (start, end) of the keyword in the TED talk. 
    
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
    Computes a vector of labels; for each tuple (start_window, end_window) in `windows_indexes` if there exists a keyword occurence during that window, 
    assign to it `kw_label`, otherwise assign the non keyword label.
    
    Args:
    windows_indexes: list of tuples (start_window, end_window) of overlapping window indexes
    keyword_occurences: list of occurences (start_kw, end_kw) of the keyword with label kw_label
    kw_label: label associated to the keyword in sentence
    non_kw_label: non keyword label
    
    Returns:
    kw_gt_vector: list of labels; eg. if kw_gt_vector[i] = kw_label this means that the window at windows_indexes[i], contains at least one occurence of the keyword.
    
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
    Given 2 vectors returned by the function `pre_gt_vector()` merge both of them element-wise following these rules:
    - non_keyword_label + non_keyword_label -> non_keyword_label
    - non_keyword_label + (any) keyword_label -> keyword_label
    - keyword_label + (different) keyword_label -> undefined_label
    - keyword_label + (same) keyword_label -> keyword_label

    Args:
    gt_vector1: vector returned by function `pre_gt_vector()`
    gt_vector2: vector returned by function `pre_gt_vector()`
    non_kw_label: non keyword label
    undefined_label: label assigned in case we try to merge 2 different keywords
    
    Returns: 
    merged_vector: vector resulting from the merger of gt_vector1 and gt_vector2 
    
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



def extract_gt_vector(windows_indexes, path2json_TED_talk, keywords, fs, undefined_label=-1, is_from_train=False, verbose=1):
    """
    Computes the ground truth vector of the TED talk in `path2json_TED_talk`; overlapping windows over the TED talk are extracted 
    and labels are assigned to each one of these frames (non_keyword_label if the frame doesn't contain any keyword, the label of 
    the keyword if the frame contains only one keyword, and `undefinied_label` if more than one keywords is present).

    Args:
    windows_indexes: list of tuples (start_window, end_window) of overlapping window indexes
    path2json_TED_talk: path to the json file which contains the aligned transcript of the TED talk
    keywords: list of keywords on which the models were trained on
    fs: sampling frequency (Hz)
    is_from_train: bool, is the TED talk from the train set (json file are constructed slightly differently)
    
    Returns:
    gt_vector: numpy array of labels.
    
    """   
    non_keyword_label = len(keywords)
    gt_vector = [non_keyword_label] * len(windows_indexes)
    
    for kw in (tqdm(keywords) if verbose else keywords):
        keyword_occurences = extract_keyword_occurences(path2json_TED_talk, kw, fs, is_from_train)
        kw_gt_vector = pre_gt_vector(windows_indexes, keyword_occurences, non_keyword_label, kw_label=keywords.index(kw))
        gt_vector = merge_gt_vectors(gt_vector, kw_gt_vector, non_kw_label=non_keyword_label, undefined_label=-1)
        
    return np.array(gt_vector)



def probability_smoothing(y_pred, w_smooth=30):
    """
    Smooth out the probabilities returned by the model. For each frame, the probability for a given label 
    is the average probabilities of the current and previous `w_smooth` frames for that label.
    
    Args:
    y_pred: matrix of probabilities returned by the model
    w_smooth: number of previous windows to take into account
    
    Returns:
    y_pred_smoothed: smoothed version of the probability matrix y_pred.
    """   
    
    rows, cols = y_pred.shape
    y_pred_smoothed = np.zeros((rows, cols))
    for j in tqdm(range(rows)):
        for i in range(cols):
            h_smooth = max(0, j - w_smooth)
            prev_prob = list(y_pred[h_smooth:(j+1),i])
            y_pred_smoothed[j, i] =  ( 1/len(prev_prob) ) * sum(prev_prob)
            
    return y_pred_smoothed



def reduce_false_alarms(y_pred):
    """
    We should detect several times the keyword when sliding the window over its occurence. Thus predicting a single or 2 keyword label on consecutive sliding frames
    would suggest having a false alarm. When having such scenarios, we assign the non_keyword label to it.    
    
    Args:
    y_pred: matrix of probabilities returned by the model
    
    Returns:
    y_pred_modified: modified version of the probability matrix y_pred.
    """   
    rows, cols = y_pred.shape
    y_pred_modified = y_pred.copy()
    non_keyword_label = cols - 1
    for i in tqdm(range(1, rows - 1)):
        for j in range(cols):
            
            # Case 1: predicting single keyword label on consecutive frames
            cond_1 = ( j != non_keyword_label ) and ( y_pred[i, j] >= 0.5 )
            cond_2 = ( y_pred[i-1, j] < 0.5 ) and ( y_pred[i+1, j] < 0.5 )
            if cond_1 and cond_2:
                y_pred_modified[i, non_keyword_label] += y_pred_modified[i, j]
                y_pred_modified[i, j] = 0
                
           # Case 2: predicting 2 consecutive keyword label on consecutive frames
            cond_1 = (i+2 < rows ) and ( j != non_keyword_label ) and ( y_pred[i, j] >= 0.5 ) \
                     and (y_pred[i+1, j] >= 0.5) and (y_pred[i-1, j] < 0.5) and (y_pred[i+2, j] < 0.5)
                
            cond_2 = (i-2 >= 0) and ( j != non_keyword_label ) and ( y_pred[i, j] >= 0.5 ) and (y_pred[i-1, j] >= 0.5) \
                     and (y_pred[i+1, j] < 0.5) and (y_pred[i-2, j] < 0.5)
            
            if cond_1:
                y_pred_modified[i, non_keyword_label] += y_pred_modified[i, j]
                y_pred_modified[i+1, non_keyword_label] += y_pred_modified[i+1, j]
                
                y_pred_modified[i, j] = 0
                y_pred_modified[i+1, j] = 0
            
            if cond_2:
                y_pred_modified[i, non_keyword_label] += y_pred_modified[i, j]
                y_pred_modified[i-1, non_keyword_label] += y_pred_modified[i-1, j]
                
                y_pred_modified[i, j] = 0
                y_pred_modified[i-1, j] = 0            
            
    return y_pred_modified



def segment_integration(y_prediction, y_prob, y_gt, non_keyword_label, segment_size=5, undefined_label=-1):
    """
    Integration of the frame level predictions into a segment level.
    
    Args:
    y_prediction: frame level prediction (argmax of matrix of probabilities)
    y_prob: frame level confidence score (max probability among labels)
    y_gt: frame level ground_trut
    segment_size: number of frames to take into account

    Returns:
    segmented_predictions: segmented level prediction.
    segmented_groundTruth: segmented level ground truth.
    """   
    
    nb_frames = len(y_prediction)
    nb_segments = int(nb_frames / segment_size)
    start = 0
    
    segmented_predictions = []
    segmented_scores = []
    segmented_groundTruth = []
    
    for i in range(nb_segments):
        
        if i == (nb_segments - 1):
            y_pred_seg = y_prediction[start:]
            y_prob_seg = y_prob[start:]
            y_gt_seg = y_gt[start:]
        else:
            y_pred_seg = y_prediction[start:start + segment_size]
            y_prob_seg = y_prob[start:start + segment_size]
            y_gt_seg = y_gt[start: start + segment_size]
        
        unique, counts = np.unique(y_gt_seg, return_counts=True) 
        nb_unique = len(unique)
        
        # Merge the Ground Truth frames
        if nb_unique == 1:
            segmented_groundTruth.append(unique[0])
        elif nb_unique == 2 and non_keyword_label in unique:
            keyword_label = unique[unique != non_keyword_label ][0]
            if counts[np.where(unique==keyword_label)] > 1:
                segmented_groundTruth.append(keyword_label)
            else:
                segmented_groundTruth.append(non_keyword_label)
        else:
            segmented_groundTruth.append(undefined_label)
        
        # Merge predicted frames into segments
        
        #unique, counts = np.unique(y_pred_seg, return_counts=True)
        #if max(counts) > 2:
        #    seg_label = unique[np.argmax(counts)]    
        #else:
        #    seg_label = non_keyword_label
        
        counts = np.bincount(y_pred_seg)
        seg_label = np.argmax(counts)
        segmented_predictions.append(seg_label)
        
        # Merge confidence scores by taking mean prob of assigned label
        segmented_scores.append(np.mean(y_prob_seg[y_pred_seg == seg_label]))
        
        start = start+segment_size
        
    return np.array(segmented_predictions), np.array(segmented_scores), np.array(segmented_groundTruth)


def segments_pruning(seg_preds, kw_label, window_dur=1.0, shift=0.1, segment_size=5):
    
    kw_occurences = []
    n_segs = len(seg_preds)
    kw_occ = []
    
    # First round of pruning
    for i, segment in enumerate(seg_preds):
        if segment == kw_label and i < n_segs - 1:
            start = shift * segment_size * i
            end = start + window_dur + (segment_size - 1) * shift
            kw_occ.append((start, end, i))
        else:
            if  len(kw_occ)>0:
                kw_occurences.append(kw_occ[0])
                kw_occ = []                
    if  len(kw_occ)>0:
        kw_occurences.append(kw_occ[0])   
       
    # Second round of pruning
    temp_indices = []
    for i,kw_occ in enumerate(kw_occurences[:-1]):
        curr_start, curr_end = kw_occ[0], kw_occ[1]
        next_start, next_end = kw_occurences[i+1][0], kw_occurences[i+1][1]
        
        if next_start < curr_end:
            temp_indices.append(i+1)
            
    kw_occ_seg_idx = [kw_occurences[i][2] for i in range(len(kw_occurences)) if i not in temp_indices]
    
    return kw_occ_seg_idx
            
    