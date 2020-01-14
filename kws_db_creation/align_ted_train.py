'''
Force alignment script for the training set of TEDLIUM's first release. 
For each TED talk, the results are saved in a single json file.
'''

import multiprocessing
import os
import sys
import gentle
import copy
import json
from joblib import Parallel, delayed
from tqdm import tqdm

disfluencies = set(['<sil>', '{SMACK}', '{NOISE}', '{BREATH}', '{UH}', '{COUGH}', '{UM}'])

def write_json(transcript, audiofile, offset, duration, output,
               nthreads=1, disfluency=False, conservative=False):
    
    """
    Force alignments of `audiofile` trimmed at (`offset`, `offset` + `duration`) with `transcript`. 
    The alignment is done at the sentence level since the transcriptions of the TED talks are discontinuous (some parts of audio aren't transcribed).
    The results are saved in a json file inside a folder named after the TED speaker where the audio comes from (aka the name of the corresponding .sph / .wav file).
    
    Args:
    transcript: string containing the transcribed text
    audiofile: path to the .wav audiofile
    offset: float indicating where the sentence starts
    duration: float indicating the length of the sentence (in seconds)
    output: path where results are saved
    nthreads: number of alignment threads (set to 1 if you are parallelizing this function)
    disfluency: bool, include disfluencies (uh, um) in alignment
    conservative: bool, conservative alignment 
    """
    
    #print("Processing audio file ", audiofile)
    resources = gentle.Resources()

    with gentle.resampled(audiofile, offset=offset, duration=duration) as wavfile:
        #print("Starting alignment")
        aligner = gentle.ForcedAligner(resources, transcript, nthreads=nthreads, disfluency=disfluency, 
                                       conservative=conservative, disfluencies=disfluencies)
        result = aligner.transcribe(wavfile)
    
    ted_speaker = output[:-5].split('/')[-1]
    
    if not os.path.exists(output[:-5]):
        os.mkdir(output[:-5])
    
    output = os.path.join(output[:-5], ted_speaker + '*' + str(offset) + '*.json')
    fh = open(output, 'w', encoding="utf-8") 
    fh.write(result.to_json(indent=2))
    fh.close()
    print("Output written to %s" % (output))

def group_sentences(ted_speaker, path2jsons='/aimlx/Datasets/TEDLIUM_release-3/data/json/', path2output='/aimlx/Datasets/TEDLIUM_release-3/data/final_json/'):
    """
    For a given TED speaker, merge all the json files created into a single one. 
    The json file created represent the alignment of the transcript with the TED talk at the word level.
    
    Args:
    ted_speaker: folder name containing the json files
    path2jsons: path to the ted_speaker's folders (each containing the json files)
    path2output: path to folder where json file are saved

    """

    print("Merging sentences for TED talk of {ts}".format(ts=ted_speaker))
    
    # Compute a sorted list of offsets, representing the beginning of each sentence in the audio file of the `ted_speaker`
    path = os.path.join(path2jsons, ted_speaker) 
        
    offsets = [float(name.split('*')[1]) for name in os.listdir(path) if not name.startswith('.')]
    offsets.sort()
    
    grouped_json = {}
    grouped_json['transcript'] = ''
    grouped_json['words'] = {}
    
    for offset in offsets:
        filename = ted_speaker + '*' + str(offset) +'*.json'

        with open(os.path.join(path2jsons, ted_speaker, filename)) as json_file:
            data = json.load(json_file)
            
            if 'transcript' in data.keys():
    
                grouped_json['transcript'] = grouped_json['transcript'] + ' ' + data['transcript']
                nb_words = len(grouped_json['words'])

                for i, word in enumerate(data['words']):
                    grouped_json['words'][i + nb_words] = copy.deepcopy(data['words'][i])

                    if not grouped_json['words'][i + nb_words]['case'] == 'not-found-in-audio':
                        grouped_json['words'][i + nb_words]['start'] += offset
                        grouped_json['words'][i + nb_words]['end'] += offset
                    
        #os.remove(os.path.join(path2jsons, ted_speaker, filename))
    
    output = os.path.join(path2output, ted_speaker + '.json')
    with open(output, 'w') as fp:
        json.dump(grouped_json, fp)

        
def get_ted_speakers(path2jsons='/aimlx/Datasets/TEDLIUM_release-3/data/json/'):
    """
    Fetching the name of all the TED talks 
    
    Args:
    path2jsons: path to the ted_speaker's folders (each containing the json files)
    
    Returns:
    ted_speakers: list of all TED talks filenames. 
    
    """
    ted_speakers = [name for name in os.listdir(path2jsons) if not name.startswith('.') and os.path.isdir(os.path.join(path2jsons, name))]
    return ted_speakers
    
#def group_all_sentences(path2jsons='/aimlx/Datasets/TEDLIUM_release1/train/json/'):
#    """
#    For all TED speakers, merge the json files. 
#    
#    Args:
#    path2jsons: path to the ted_speaker's folders (each containing the json files)
#    
#    """
#    ted_speakers = [name for name in os.listdir(path2jsons) if not name.startswith('.') and os.path.isdir(os.path.join(path2jsons, name))]
#    for ted_speaker in tqdm(ted_speakers):
#        path = os.path.join(path2jsons, ted_speaker) 
#        offsets = [float(name.split('*')[1]) for name in os.listdir(path) if not name.startswith('.')]
#        offsets.sort()
#        group_sentences(ted_speaker, offsets, path2jsons)
        
        
def get_transcript(path2transcript):
    """
    For a given transcription file, create a list of tuples (start, duration, label) where: 
    - start is the offset (the beginning) of the sentence in the audio file
    - duration is the length of the sentence (in seconds)
    - label is the transcribed text (disfluencies such as {SMACK} and {COUGH} are removed)
    
    Args:
    path2transcript: path to .stm transcription file
    
    Returns:
    sentences: list of tuples (start, duration, label) sorted according to `start`.
    
    """
    sentences = []
    with open(path2transcript, 'rt') as f:
        records = f.readlines()
        for record in records:
            fields = record.split()
            label = fields[6:-1]
            start, end = float(fields[3]), float(fields[4])
            
            #Remove disfluencies
            label = [word for word in label if '{' not in word and '<' not in word]
            label = [word[:-3] if word.endswith(')') else word for word in label]
            label = " ".join(label)
            sentences.append((start, end-start, label))
            
    sentences = sorted(sentences, key=lambda x: x[0])    
    return sentences   
    
    
def get_files(path='/aimlx/Datasets/TEDLIUM_release1/train/wav'):
    """
    Create 2 lists of same length;
    - the second contains the information (start, duration, label) of a given sentence in a given transciption file
    - the first contains the path to the corresponding audiofile
    
    For alignment purposes, the first list contains repetitions (each path is repeated `n_sentences` times
    where n_sentences is the number of sentencs in the transcription file). This is useful when parallelizing with joblib
    
    Args:
    path: path to the folder containing the .wav audio files
    
    Returns:
    audiofiles: list of paths to the .wav audio file. 
    all_sentences: list of tuples (start, duration, label) for all sentences in all audiofiles.
    
    """
    audiofiles = []
    all_sentences = []
    invalid_stms = []
    nb_audio_file = 0
    
    for _, _, files in os.walk(path):
        for file in files[1647:]:
            if file.endswith('.wav'):
                path2txt = os.path.join(path, file).replace('wav', 'stm')
                try:
                    sentences = get_transcript(path2transcript=path2txt)                 
                    n_sentences = len(sentences)
                    all_sentences.extend(sentences)
                    audiofiles.extend([os.path.join(path, file)] * n_sentences)
                    nb_audio_file += 1
                except:
                    invalid_stms.append(path2txt)
    return audiofiles, all_sentences, invalid_stms

#audiofiles, all_sentences, invalid_stms = get_files(path='/aimlx/Datasets/TEDLIUM_release-3/data/wav')
#print(len(audiofiles))
#print(len(all_sentences))
#print(invalid_stms)


#Create json files for each sentence
#Parallel(n_jobs=12, prefer="threads")(delayed(write_json)(sentence[2], audiofile, offset=sentence[0], duration=sentence[1], output=audiofile.replace('wav','json'))
#                                        for audiofile, sentence in zip(audiofiles, all_sentences))

#Merge json files for each TED talk
#ted_speakers = get_ted_speakers()
#Parallel(n_jobs=11, prefer="threads")(delayed(group_sentences)(ted_speaker) for ted_speaker in ted_speakers)
