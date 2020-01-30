import os
import numpy as np
from tqdm import tqdm
import scipy.io.wavfile as wav
from postprocessing import generate_windows_indexes, extract_gt_vector, mfcc_evaluation_set


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