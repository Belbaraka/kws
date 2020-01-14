import os, re
import numpy as np
import hashlib
from tqdm import tqdm
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import librosa


def augment_with_noise(signal, noise_factor=1.0):
    """  
    Adds white gaussian noise to the signal

    Args:
    signal: signal which contains a keyword or a non-keyword sample
    noise_factor: standard deviation of the noise (default to 1)

    Returns:
    augmented_data : noisy version of the orginal signal.
    """     
    
    noise = np.random.randn(len(signal))
    augmented_data = signal + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(signal[0]))
    return augmented_data

def augment_with_volume(signal):
    """  
    Multiply signal by a random number between 1.5 and 3.

    Args:
    signal: signal which contains a keyword or a non-keyword sample

    Returns:
    Amplified version of the orginal signal.
    """ 
    
    vol_factor = np.random.uniform(low=1.5,high=3)
    return vol_factor * signal

def pitch_changing(signal, fs, pitch_factor=4):
    """
    Change pitch of signal.

    Args:
    signal: signal which contains a keyword or a non-keyword sample
    fs: sampling frequency
    
    Returns:
    New version of the orginal signal.
    """ 
    return librosa.effects.pitch_shift(signal, fs, n_steps=pitch_factor)

def compute_mfcc(signal, fs, threshold=1.0, num_features=40, is_kw=True, add_noise=False, manipulate_pitch=False, add_volume=False): 
    """
    Computes the MFCC features. The signal is first passed through a low pass filter with frequencies 20Hz-4000Hz. 
    Various data augmentation techniques can be used on the signal before computing the MFCC features.
    
    Args:
    signal: signal which contains a keyword or a non-keyword sample
    fs: sampling frequency (usually 16kHz)
    threshold: fixed size window (in seconds) over which the MFCCs are computed
    num_features: number of MFCCs
    add_noise: bool, augment with noise 
    manipulate_pitch: bool, augment with pitch change
    add_volume: bool; augment with volume 
    
    Returns:
    all_features: list of features matrices; if for examples only add_noise is enabled, a list of 2 features matrices is returned
                  one for the orignal signal and the other for the noisy version of the signal.
    """       
    dur = len(signal)
    if dur < int(threshold * fs):
        zeros = np.zeros((int(threshold * fs) - dur, 1)).reshape(-1,)
        signal = np.concatenate((signal, zeros), axis=0)
    else:
        signal = signal[:int(threshold * fs)]
    
    
    features = mfcc(signal, samplerate=fs, winlen=0.030, winstep=0.01, numcep=num_features,
                    lowfreq=20, highfreq=4000, appendEnergy=False, nfilt=num_features)

    all_features = []
    all_features.append(features)
    
    if add_noise and is_kw:
        noisy_signal = augment_with_noise(signal, noise_factor=1.0)
        noisy_features = mfcc(noisy_signal, samplerate=fs, winlen=0.030, winstep=0.01, numcep=num_features,
                                lowfreq=20, highfreq=4000, appendEnergy=False, nfilt=num_features)

        all_features.append(noisy_features)

    if manipulate_pitch and is_kw:
        changed_pitch_signal = pitch_changing(signal.astype(float), fs, pitch_factor=4)
        changed_pitch_features = mfcc(changed_pitch_signal, samplerate=fs, winlen=0.030, winstep=0.01, numcep=num_features,
                                      lowfreq=20, highfreq=4000, appendEnergy=False, nfilt=num_features)

        all_features.append(changed_pitch_features)
    
    if add_volume and is_kw:
        louder_signal = augment_with_volume(signal)
        louder_signal_features = mfcc(louder_signal, samplerate=fs, winlen=0.030, winstep=0.01, numcep=num_features,
                                      lowfreq=20, highfreq=4000, appendEnergy=False, nfilt=num_features)

        all_features.append(louder_signal_features)        

    return all_features
    

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
    
    MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
    
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


def generate_sets(filenames, keywords, frame_size=0.6, n_mfcc=40, validation_percentage=10, testing_percentage=10, add_noise=False, manipulate_pitch=False, add_volume=False):
    """
    Computes the datasets used for training, validation and testing.

    Args:
    filenames: list of paths to the wav signals of the keyword and non-keyword samples 
    keywords: list of keywords
    frame_size: size in second of frame over which MFCCs are computed
    n_mfcc: number of MFCC features to compute
    validation_percentage: percentage of data used for validation
    testing_percentage: percentage of data used for testing
    add_noise: bool, augment with noise 
    manipulate_pitch: bool, augment with pitch change
    add_volume: bool; augment with volume 
    
    Returns:
    training, validation, testing: lists of features matrices and their respective label, each element in these lists is a tuple (feature_matrix, label)
    """  
    
    non_keyword_label = len(keywords)

    training, validation, testing = [], [], []

    for filename in tqdm(filenames):
        _, signal = wav.read(filename)
        kw = filename.split('/')[-2]
        
        is_kw = kw in keywords
        all_features = compute_mfcc(signal, fs=16000, threshold=frame_size, num_features=n_mfcc, is_kw=is_kw, 
                                    add_noise=add_noise, manipulate_pitch=manipulate_pitch, add_volume=add_volume)
        
        if kw in keywords:
            label = keywords.index(kw)
        else:
            label = non_keyword_label
            
        grp = which_set(filename, validation_percentage, testing_percentage)
        
        X_y = [(feats, label) for feats in all_features]
        
        if grp is 'training':
            training.extend(X_y)
        elif grp is 'validation' :
            validation.extend(X_y)
        else:
            testing.extend(X_y)
    
    return training, validation, testing

def get_X_y(grp, xdim, num_features=40):
    """
    Shapes the data (features matrices, labels) to the correct format.
    
    Args:
    grp: list of tuples (feature_matrix, label)
    xdim: row dimension of the feature matrix
    num_features: number of MFCCs (column dimension of feature matrix)
    
    Returns:
    X: feature matrix with shape (xdim, num_features)
    y: numpy array of labels
    """      
    X, y = zip(*grp)
    X = list(map(lambda x: x.reshape(xdim, num_features, 1), X))
    return np.array(X, dtype=np.float32).reshape(-1, xdim, num_features, 1), np.array(y, dtype=np.float32).reshape(-1,1)