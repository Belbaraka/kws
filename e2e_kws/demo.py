import numpy as np
import warnings
warnings.filterwarnings('ignore')
from random import sample
import os
import scipy.io.wavfile as wav
import numpy as np
from demo import *
from postprocessing import compute_mfcc_frames, probability_smoothing, reduce_false_alarms
import keras
import IPython
from models import *
import io


kws_sets = [['people', 'because', 'think', 'world'], ['something', 'different', 'actually','important'], 
            ['another', 'percent', 'problem', 'technology'], ['information', 'experience', 'government', 'computer']]

path2_models = {1: {'cnn_parada': 'set1_models/cnn_parada_set1_5_epochs_new_db.h5', 'dnn': 'set1_models/dnn_set1_5_epochs_new_db.h5'},
                2: {'cnn_parada': 'set2_models/cnn_parada_set2_4_epochs_new_db.h5', 'dnn': 'set2_models/dnn_set2_4_epochs_new_db.h5'},
                3: {'cnn_parada': 'set3_models/cnn_parada_set3_4_epochs_new_db.h5', 'dnn': 'set3_models/dnn_set3_4_epochs_new_db.h5'},
                4: {'cnn_parada': 'set4_models/cnn_parada_set4_3_epochs_new_db.h5', 'dnn': 'set4_models/dnn_set4_3_epochs_new_db.h5'}}

path2data = '/Users/Belbaraka/Desktop/Swisscom/Thesis/Datasets/demo'


def frames2segments(y_prediction, non_keyword_label, segment_size=5):
    """
    Integration of the frame level predictions into a segment level.
    
    Args:
    y_prediction: frame level prediction (argmax of matrix of probabilities)
    segment_size: number of frames to take into account

    Returns:
    segmented_predictions: segmented level prediction.
    """   
    
    nb_frames = len(y_prediction)
    nb_segments = int(nb_frames / segment_size)
    start = 0
    
    segmented_predictions = []
    segmented_groundTruth = []
    
    for i in range(nb_segments):
        
        if i == (nb_segments - 1):
            y_pred_seg = y_prediction[start:]
        else:
            y_pred_seg = y_prediction[start:start + segment_size]
        
        # Merge predicted frames into segments
        unique, counts = np.unique(y_pred_seg, return_counts=True)
        if max(counts) > 2:
            seg_label = unique[np.argmax(counts)]    
        else:
            seg_label = non_keyword_label
        segmented_predictions.append(seg_label)
        
        start = start+segment_size 
    return np.array(segmented_predictions)


def extract_keyword_occurences(seg_preds, kw_label, window_dur=1.0, shift=0.1, segment_size=5):
    
    kw_occurences = []
    n_segs = len(seg_preds)
    kw_occ = []
    
    
    # First round of pruning
    for i, segment in enumerate(seg_preds):
        if segment == kw_label and i < n_segs - 1:
            start = shift * segment_size * i
            end = start + window_dur + (segment_size - 1) * shift
            kw_occ.append((start, end))
        else:
            if  len(kw_occ)>0:
                kw_occurences.append(sample(kw_occ, k=1)[0])
                kw_occ = []                
    if  len(kw_occ)>0:
        kw_occurences.append(sample(kw_occ, k=1)[0])   
       
    # Second round of pruning
    temp_indices = []
    for i,kw_occ in enumerate(kw_occurences[:-1]):
        curr_start, curr_end = kw_occ[0], kw_occ[1]
        next_start, next_end = kw_occurences[i+1][0], kw_occurences[i+1][1]
        
        if next_start < curr_end:
            temp_indices.append(i+1)
    
    return [kw_occurences[i] for i in range(len(kw_occurences)) if i not in temp_indices]


def keywords_found(keywords, predicted_segments, window_dur=1.0, shift=0.1, segment_size=5):
    labels, _ = np.unique(predicted_segments, return_counts=True)
    non_kw_label = len(keywords)
    all_kw_occurences = []
    kw_occurences = []
    
    for label in labels:
        if label != non_kw_label:
            kw_occurences = extract_keyword_occurences(predicted_segments, label, window_dur, shift, segment_size)
            print('Found ' + str(len(kw_occurences)) + ' occurence(s) of the keyword: '+ keywords[label] )
        
        all_kw_occurences.append((label, kw_occurences))
    return all_kw_occurences


from ipywebrtc import AudioStream, AudioRecorder, CameraStream

def set_up_recorder():
    camera = CameraStream(constraints=
                      {'facing_mode': 'user',
                       'audio': True
                       })
    #recorder = AudioRecorder(stream=camera, codecs='opus')
    recorder = AudioRecorder(stream=camera)
    
    return recorder

from io import BytesIO
from pydub import AudioSegment

def process_recording(user_id, recorder, _set): 
    
    recorder.save('record.webm')
    sound = AudioSegment.from_file('record.webm', codec="opus")
    sound = sound.set_frame_rate(16000)
    sound = sound.set_channels(1)
    
    filename = os.path.join(path2data, 'set' + str(_set), user_id + '.wav') 
    sound.export(filename, format='wav')
    fs, sig = wav.read(filename)
    
    return fs, sig

def spot_keywords(user_id, _set, recorder, model_type='dnn', shift=0.001, w_smooth=3, segment_size=5, path2wav=None):
    keywords = kws_sets[_set - 1]
    path2_model = path2_models[_set][model_type]
    model = keras.models.load_model(path2_model)

    num_features = 40
    xdim = 98
    frame_dur = 1.0
    non_keyword_label = len(keywords)
    
    print("[1/4]: Loading test audio file and computing MFCC frames")
    if path2wav is None:
        fs, sig = process_recording(user_id, recorder, _set)
    else:
        fs, sig = wav.read(path2wav)
    sig_frames = compute_mfcc_frames(sig, fs, xdim=xdim, shift=shift, num_features=num_features, verbose=1)
    
    print("[2/4]: Frame-level predictions")
    y_pred = model.predict(sig_frames, verbose=1)
    
    print("[3/4]: Post-processing and segment-level integration")
    y_pred_modified = reduce_false_alarms(y_pred)
    y_pred_smoothed_post = probability_smoothing(y_pred_modified, w_smooth=w_smooth)
    y_prediction_smoothed_post = np.argmax(y_pred_smoothed_post, axis=1) 
    predicted_segments = frames2segments(y_prediction_smoothed_post, non_keyword_label, segment_size)
    
    print("[4/4]: Extracting keyword occurrences\n")
    
    print("*************************************")
    
    all_kw_occurrences = keywords_found(keywords, predicted_segments, window_dur=frame_dur, 
                                        shift=shift, segment_size=segment_size)
    if len(all_kw_occurences) == 0:
        print('No keywords found')
        
    print("*************************************")

    return all_kw_occurrences, sig, fs
    
    
def listen2kw(kw, all_kw_occurrences, _set, fs=16000):
    keywords = kws_sets[_set - 1]
    kw_occurrences = list(filter(lambda x: x[0] == keywords.index(kw), all_kw_occurrences))[0][1]
    
    return iter(kw_occurrences)
    