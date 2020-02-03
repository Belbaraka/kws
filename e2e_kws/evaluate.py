'''
Script that performs the evaluation of models on the test TED talks
'''
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io.wavfile as wav
import argparse
from postprocessing import *
from scoring import *
import warnings
warnings.filterwarnings('ignore')
import keras


if __name__ == "__main__":
    
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    
    # Required args
    parser.add_argument('-kw', '--keywords', type=str, nargs='+',
                        help='list of keywords to spot', required=True)

    parser.add_argument('-p2w', '--path2wav', type=str, default='/aimlx/Datasets/TEDLIUM_release-3/data/wav',
                        help='path to the .wav TED talks')    

    parser.add_argument('-p2j', '--path2json', type=str, default='/aimlx/Datasets/TEDLIUM_release-3/data/final_json',
                        help='path to json files which contains word level alignment of ted talks')    
    
    parser.add_argument('-p2ted', '--path2tedtalks', type=str, default='/aimlx/kws/e2e_kws/test_data/test_ted_talks_50.npy',
                        help='path to the saved list of ted talks kept for testing (array of filenames)')       
    
    parser.add_argument('-p2m', '--path2model', type=str, default='/aimlx/kws/e2e_kws/demo/models/set1_models/dnn_set1_5_epochs.h5',
                        help='path to the model trained to spot the keywords given above') 
    
    
    # Optionnal args
    parser.add_argument('-wdur', '--wduration', type=float, default=1.0,
                        help='duration of the window taken from speech')
    
    parser.add_argument('-s', '--shift', type=float, default=0.1, 
                        help='amount in seconds by which windows of speeches taken are shifted ')
    
    parser.add_argument('-ws', '--wsmooth', type=int, default=3,
                        help='number of previous label probabilities to take into account to smooth out predictions')
    
    parser.add_argument('-xdim', '--xdim', type=int, default=98,
                        help='dimension along the x-axis of MFCC frames')  
    
    parser.add_argument('-nfeats', '--nfeatures', type=int, default=40,
                        help='number of mfcc features computes')    
    
    
    args = parser.parse_args()
    
    #Required args
    _keywords = args.keywords
    _path2wav = args.path2wav
    _path2json = args.path2json
    _path2test_talks = args.path2tedtalks
    _path2model  = args.path2model
    
    # Optional args
    _window_dur = args.wduration
    _shift = args.shift
    _w_smooth = args.wsmooth
    _xdim = args.xdim
    _nfeatures = args.nfeatures
    
    
    test_ted_talks = np.load(_path2test_talks, allow_pickle=True)
    
    print("[1/5] : Extracting test frames and ground truth labels from test ted talks")
    test_frames, test_gt_vector, T = prepare_test_ted_talks(test_ted_talks[1:2], shift=_shift, window_dur=_window_dur, xdim=_xdim, num_features=_nfeatures, keywords=_keywords,
                                                     path2wav=_path2wav, path2json=_path2json)
    
    model = keras.models.load_model(_path2model)
    
    print("[2/5] : Predictions on test frames")
    y_pred = model.predict(test_frames, verbose=1)
    
    print("[3/5] : Post processing of predictions; probability smoothing and false alarm reduction")
    y_pred_modified = reduce_false_alarms(y_pred)
    y_pred_smoothed_post = probability_smoothing(y_pred_modified, w_smooth=3)
    y_prediction_smoothed_post = np.argmax(y_pred_smoothed_post, axis=1) 
    
    print("[4/5] : Segment level integration of prediction")
    seg_preds, seg_scores, seg_gt = segment_integration(y_prediction_smoothed_post, np.max(y_pred_smoothed_post, axis=1),
                                                        test_gt_vector, non_keyword_label=len(_keywords), segment_size=5, undefined_label=-1)
    
    print("[5/5] : Computing FOM results")
    
    df_results = pd.DataFrame(columns=['KeyWord', '#Hits', '#FA\'s', '#Actual', 'FOM', 'Recall', 'Precision'])
    for i in range(len(_keywords)):
        hits, false_alarms, nb_TP, fom = fom_result(seg_preds, seg_scores, seg_gt, keyword_label=i, T=T, window_dur=1.0, shift=0.1, segment_size=5)
        if hits == 0:
            recall, precision = 0, 0
        else:
            recall, precision = round(hits/nb_TP, 2), round(hits/(hits+false_alarms), 2)
        df_results = df_results.append({'KeyWord': _keywords[i], '#Hits': hits, '#FA\'s': false_alarms, '#Actual': nb_TP, 'FOM': round(fom, 2), 
                                        'Recall': recall, 'Precision': precision}, ignore_index=True)
    
    df_results.to_pickle("df_results.pkl")
    print("FOM Resuls")
    print(df_results)
    
    print('\nAbove dataframe saved in current working directory')
    