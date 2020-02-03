import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io.wavfile as wav
import argparse
from postprocessing import *
import warnings
warnings.filterwarnings('ignore')


def prepare_test_ted_talks(test_ted_talks, keywords, shift=0.1, window_dur=1.0, xdim=98, num_features=40,
                           path2wav='/aimlx/Datasets/TEDLIUM_release-3/data/wav', path2json='/aimlx/Datasets/TEDLIUM_release-3/data/final_json',
                           path2stm='/aimlx/Datasets/TEDLIUM_release-3/data/stm'):
    '''
    Cleans the test TED talks by removing part of the signal on which there isn't speech (music intro, clapping,...), computes MFCC frames and ground truth labels.

    Args:
    test_ted_talks: list of TED talks reserved for testing
    keywords: list of keywords to be extracted from TED talks
    shift: amount (in seconds) by which windows are shifted
    window_dur: window/frame duration (in seconds)
    xdim: row dimension of the feature matrix
    num_features: number of MFCCs (column dimension of feature matrix)
    path2wav: path to wav TED talks
    path2json: path to json files containing the word level alignements
    path2stm: path to .stm transcription containing sentence level alignements
    
    Returns:
    test_frames: List of the windows extracted from the signal.
    test_gt_vector: List ground truth labels
    T: duration of test speech in hours
    '''
    test_gt_vector = np.array([])
    test_frames = np.array([])
    T = 0
    
    for ted_talk in tqdm(test_ted_talks):
        fs, sig = wav.read(os.path.join(path2wav, ted_talk + '.wav'))
        signal_duration = len(sig)
        #T += (signal_duration / fs) / 3600 
        
        windows_indexes = generate_windows_indexes(signal_duration, fs, window_dur=window_dur, shift=shift)
        path2stm_TED_talk = os.path.join(path2stm, ted_talk + '.stm')
        
        # list of tuple (start_sentence, end_sentence) 
        aligned_sentences = []
        with open(path2stm_TED_talk, 'rt') as f:
            records = f.readlines()
            for record in records:
                fields = record.split()
                start, end = int(fs*float(fields[3])), int(fs*float(fields[4]))
                T += ( (end - start) / fs ) / 3600 #duration of sentence in hours
                aligned_sentences.append((start, end))
         
        filtered_w_indexes = list(filter(lambda x: any( (x[0] >= s[0] and x[1] <= s[1]) for s in aligned_sentences), windows_indexes))        
        
        sig_frames = mfcc_evaluation_set(signal=sig, fs=fs, windows=filtered_w_indexes, xdim=xdim, num_features=num_features, verbose=0)
        
        if len(test_frames) == 0:
            test_frames = sig_frames
        else:
            test_frames = np.concatenate([test_frames, sig_frames], axis=0)
        
        path2json_TED_talk = os.path.join(path2json, ted_talk + '.json')
        gt_vector = extract_gt_vector(filtered_w_indexes, path2json_TED_talk, keywords=keywords, fs=fs, is_from_train=True, verbose=0)
        test_gt_vector = np.concatenate([test_gt_vector, gt_vector])

    return test_frames, test_gt_vector, T



def fom_result(seg_preds, seg_scores, seg_gt, keyword_label, T, window_dur=1.0, shift=0.1, segment_size=5, undefined_label=-1):
    """
    Computes the Figure of Merit (FOM) defined by NIST which is an upper-bound estimate on word spotting accuracy averaged over 1 to 10 false alarms per hour.
    The FOM is calculated as follows where it is assumed that the total duration of the test speech is T hours. 
    For each word, all of the spots are ranked in score order. The percentage of true hits pi 
    found before the i’th false alarm is then calculated for i = 1 . . . N + 1 where N is the first integer ≥ 10T − 0.5. 
    The FOM is then defined as : 1/(10T) *(p1 + p2 + ... + pN + ap(N+1)), where a = 10T − N is a factor that interpolates to 10 false alarms per hour

    Args:
    seg_preds: segmented level prediction.
    seg_scores: segmented level confidence scores for the label assigned 
    seg_gt: segmented level ground truth.
    keyword_label: label of of the keyword for which the FOM is calculated.
    T: duration in hours of test speech.

    Returns:
    hits: number of hits
    false_alarms: number of false alarms
    nb_TP : number of actual keyword label
    fom: figure of merit for the given keyword
    """   
    
    predicted_kw_indexes = segments_pruning(seg_preds, keyword_label, window_dur=window_dur, shift=shift, segment_size=segment_size)
    confidence_scores = seg_scores[predicted_kw_indexes]    
    
    #Rank predicted keyword indexes according to confidence score
    sorted_pred_kw_indexes = [x for _, x in sorted( zip( confidence_scores, predicted_kw_indexes), key=lambda pair: pair[0], reverse=True)]

    pred_kw_occs, gt_kw_occs = seg_preds[sorted_pred_kw_indexes], seg_gt[sorted_pred_kw_indexes]
    
    gt_kw_indexes = segments_pruning(seg_gt, keyword_label, window_dur=window_dur, shift=shift, segment_size=segment_size)
    
    N = math.ceil(10*T - 0.5)
    a = 10*T - N
    hits = sum(pred_kw_occs == gt_kw_occs)

    probabilities = [] # list of probabilities p1, p2, ...
    
    # Number of actual keywords test speech
    nb_TP = len(gt_kw_indexes)
    
    false_alarm_indices = np.where((gt_kw_occs != keyword_label) & (gt_kw_occs != undefined_label) )[0]
    false_alarms = len(false_alarm_indices)

    probabilities = []
    #print(false_alarm_indices)
    
    for i in range(0, N + 1):
        if i < len(false_alarm_indices):
            i_th_false_alarm = false_alarm_indices[i]
            tmp = gt_kw_occs[:i_th_false_alarm]
            nb_hits = sum(tmp == keyword_label)
            probabilities.append( nb_hits/nb_TP )
        else:
            nb_hits = sum(gt_kw_occs == keyword_label)
            probabilities.append( nb_hits/nb_TP )
    
    #print(probabilities)
    probabilities[-1] = a*probabilities[-1]
    fom = (1/ (10*T))* sum(probabilities)

    return hits, false_alarms, nb_TP, 100*fom




    