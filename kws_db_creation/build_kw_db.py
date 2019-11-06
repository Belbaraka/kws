'''
Builds keyword database after running force alignment on audio files
'''

import os
import json
from scipy.io import wavfile
from joblib import Parallel, delayed
from tqdm import tqdm

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


def get_jsons(path='/aimlx/Datasets/TEDLIUM_release1/dev/json'):
    """
    Get the paths of the json files containing the forced alignement's key informations

    Args:
    path: Path to folder which contains the json files

    Returns:
    List, paths to json files.
    """ 
    path2jsons = []
    for _, _, files in os.walk(path):
        for file in files:
            if file.endswith('.json') and 'checkpoint' not in file:
                path2jsons.append(os.path.join(path, file))    
                
    return path2jsons

def extract_kw_dev_test(keyword, path2jsons, path2kw_db='/aimlx/Datasets/TEDLIUM_release1/kw_db'):
    """
    Aligning the keyword with the audio files, extracting and saving it as a new .wav file in path2kw_db/keyword/{speaker_name}-{i}.wav. 
    The naming convention is {speaker_name}-{i}, where i is the occurence index of the keyword in the transcription.
    This function is designed for the test and dev set of TEDLIUM.
    
    Args:
    keyword: desired keyword to be extracted
    path2jsons: list of json files containing the forced alignement's key informations
    path2kw_db: path where to save the extracted keywords
    
    """ 
    print('Extracting keyword {kw} from all  audio files'.format(kw=keyword))
    if not os.path.exists(os.path.join(path2kw_db, keyword)):
        os.mkdir(os.path.join(path2kw_db, keyword))
        
    for path in path2jsons:
        with open(path) as json_file:
            data = json.load(json_file)
            count = 0
            for word in data['words']:
                if word['word'] == keyword and word['case'] == 'success':
                    start, end = word['start'], word['end']
                    offset = get_offset(path2transcription=path.replace('json', 'stm'))
                    fs, signal = wavfile.read(path.replace('json', 'wav'))
                    start, end = int(fs * (start + offset)), int(fs * (end + offset))
                    filename = path.split('/')[-1].split('.')[0] + '-' + str(count) + '.wav'
                    wavfile.write(os.path.join(path2kw_db, keyword, filename), data=signal[start:end], rate=fs)
                    count += 1 


                    
def extract_kw_train(keyword, path2jsons, path2kw_db='/aimlx/Datasets/TEDLIUM_release1/kw_db'):
    """
    This function is designed for the train set of TEDLIUM.

    Aligning the keyword with the audio files, extracting and saving it as a new .wav file in path2kw_db/keyword/{speaker_name}_{i}.wav. 
    The naming convention is {speaker_name}_{i}, where i is the occurence index of the keyword in the transcription.
    
    Args:
    keyword: desired keyword to be extracted
    path2jsons: list of json files containing the forced alignement's key informations
    path2kw_db: path where to save the extracted keywords
    
    """ 
    print('Extracting keyword {kw} from all  audio files'.format(kw=keyword))
    
    if not os.path.exists(os.path.join(path2kw_db, keyword)):
        os.mkdir(os.path.join(path2kw_db, keyword))
        
    for path in tqdm(path2jsons):
        with open(path) as json_file:
            data = json.load(json_file)
            count = 0
            for word_id in data['words']:
                word = data['words'][word_id]
                if word['word'] == keyword and word['case'] == 'success':
                    start, end = word['start'], word['end']
                    
                    path = path.replace('.json', '.wav')
                    path = path.replace('final_json', 'wav')
                    fs, signal = wavfile.read(path)
                    
                    start, end = int(fs * start), int(fs * end)
                    filename = path.split('/')[-1].split('.')[0] + '-' + str(count) + '.wav'
                    wavfile.write(os.path.join(path2kw_db, keyword, filename), data=signal[start:end], rate=fs)
                    count += 1 
    
# Build the keywords database
path2jsons_dev = get_jsons(path='/aimlx/Datasets/TEDLIUM_release1/dev/json')
path2jsons_test = get_jsons(path='/aimlx/Datasets/TEDLIUM_release1/test/json')
path2jsons_train = get_jsons(path='/aimlx/Datasets/TEDLIUM_release1/train/final_json')[:100]

keywords = ['people', 'because', 'vision', 'important', 'world', 'today']

Parallel(n_jobs=11)(delayed(extract_kw_train)(keyword, path2jsons_train) for keyword in keywords)