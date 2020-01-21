import os
import numpy as np
from tqdm import tqdm
import scipy.io.wavfile as wav
from postprocessing import generate_windows_indexes, extract_gt_vector, compute_mfcc_frames


def prepare_test_ted_talks(test_ted_talks, keywords, shift=0.1, window_dur=1.0, xdim=98, num_features=40,
                           path2wav='/aimlx/Datasets/TEDLIUM_release-3/data/wav', path2json='/aimlx/Datasets/TEDLIUM_release-3/data/final_json'):
    '''
    Extract the MFCC frames from the test ted talks

    Args:
    test_ted_talks: 
    keywords:
    shift: amount (in seconds) by which windows are shifted
    window_dur: window/frame duration (in seconds)
    xdim:
    num_features:
    path2wav:
    path2json:
    
    Returns:
    test_frames: List of the windows extracted from the signal.
    test_gt_vector: 
    T: duration of test speech in hours
    '''
    test_gt_vector = np.array([])
    test_frames = np.array([])
    T = 0
    
    for ted_talk in tqdm(test_ted_talks):
        fs, sig = wav.read(os.path.join(path2wav, ted_talk + '.wav'))
        signal_duration = len(sig)
        T += (signal_duration / fs) / 3600 #duration of test speech in hours
        
        sig_frames = compute_mfcc_frames(sig, fs, w_dur=window_dur, xdim=xdim, shift=shift, num_features=num_features, verbose=0)
        
        if len(test_frames) == 0:
            test_frames = sig_frames
        else:
            test_frames = np.concatenate([test_frames, sig_frames], axis=0)
        
        windows_indexes = generate_windows_indexes(signal_duration, fs, window_dur=window_dur, shift=shift)
        path2json_TED_talk = os.path.join(path2json, ted_talk + '.json')
        gt_vector = extract_gt_vector(windows_indexes, path2json_TED_talk, keywords=keywords, fs=fs, is_from_train=True, verbose=0)
        test_gt_vector = np.concatenate([test_gt_vector, gt_vector])

    return test_frames, test_gt_vector, T