import numpy as np
from random import sample

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

