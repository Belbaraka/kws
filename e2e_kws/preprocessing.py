import os, sys, re
from itertools import islice 
import numpy as np
import pandas as pd
import hashlib
import random
import math
import json
from random import sample
from tqdm import tqdm
from python_speech_features import mfcc
import scipy.io.wavfile as wav


def augment_with_noise(signal, noise_factor=1.0):
    """
    TODO : add description

    Args:
    signal: 
    noise_factor:

    Returns:
    augmented_data, paths to json files.
    """     
    
    noise = np.random.randn(len(signal))
    augmented_data = signal + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(signal[0]))
    return augmented_data


def compute_mfcc(signal, fs, threshold=1.0, num_features=40, add_noise=True): 
    """
    TODO : add description

    Args:
    signal: 
    fs:
    threshold:
    num_features:
    add_noise:
    
    Returns:
    augmented_data, paths to json files.
    """       
    dur = len(signal)
    if dur < int(threshold * fs):
        zeros = np.zeros((int(threshold * fs) - dur, 1)).reshape(-1,)
        signal = np.concatenate((signal, zeros), axis=0)
    else:
        signal = signal[:int(threshold * fs)]
    
    features = mfcc(signal, samplerate=fs, winlen=0.030, winstep=0.01, numcep=num_features,
                    lowfreq=20, highfreq=4000, appendEnergy=False, nfilt=num_features)
    
    if add_noise:
        noisy_signal = augment_with_noise(signal, noise_factor=1.0)
        noisy_features = mfcc(noisy_signal, samplerate=fs, winlen=0.030, winstep=0.01, numcep=num_features,
                    lowfreq=20, highfreq=4000, appendEnergy=False, nfilt=num_features)
        
        return features, noisy_features
    else:
        return features
    
    
    
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def which_set(filename, validation_percentage, testing_percentage):
    """
    Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_.' in a filename is
    ignored for set determination. This ensures that 'AdamSavage_2008P_5.wav' and
    'AdamSavage_2008P_0.wav' are always in the same set, for example.

    Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

    Returns:
    String, one of 'training', 'validation', or 'testing'.
    """
    
    kw = filename.split('/')[-2]
    base_name = kw + '-' + os.path.basename(filename).replace('_', '-', 1)
    # We want to ignore anything after '-' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_.*$', '', base_name).encode('utf-8')
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result


def generate_sets(filenames, keywords, validation_percentage=10, testing_percentage=10, add_noise=True):
    """
    TODO : add description

    Args:
    filenames: 
    validation_percentage:
    testing_percentage:
    add_noise:
    
    Returns:
    training: 
    validation:
    testing:
    """  
    
    non_keywords_label = len(keywords)

    training, validation, testing = [], [], []

    for filename in tqdm(filenames, position=0, leave=True):
        _, signal = wav.read(filename)
        kw = filename.split('/')[-2]

        if add_noise and (kw in keywords):
            feats, noisy_feats = compute_mfcc(signal, fs=16000, threshold=1.0, num_features=40, add_noise=add_noise)
        else:
            feats = compute_mfcc(signal, fs=16000, threshold=1.0, num_features=40, add_noise=False)
                            
        if kw in keywords:
            label = keywords.index(kw)
        else:
            label = non_keywords_label
            
        grp = which_set(filename, validation_percentage, testing_percentage)
        
        if grp is 'training':
            training.append((feats, label))
            if add_noise and (kw in keywords):
                training.append((noisy_feats, label))
        elif grp is 'validation' :
            validation.append((feats, label))
            if add_noise and (kw in keywords):
                validation.append((noisy_feats, label))            
        else:
            testing.append((feats, label))
            if add_noise and (kw in keywords):
                testing.append((noisy_feats, label))            
    
    return training, validation, testing

def get_X_y(grp, xdim, num_features=40):
    """
    TODO : add description

    Args:
    grp: 
    xdim:
    num_features:
    
    Returns:
    X: 
    y:
    """      
    X, y = zip(*grp)
    X = list(map(lambda x: x.reshape(xdim, num_features, 1), X))
    return np.array(X).reshape(-1, xdim, num_features, 1), np.array(y).reshape(-1,1)