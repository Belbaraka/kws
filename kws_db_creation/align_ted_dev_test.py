'''
Force alignment script for the dev and test sets of TEDLIUM's first release. 
For each TED talk, the results are saved in a single json file.
'''

import multiprocessing
import os
import sys
import gentle
from joblib import Parallel, delayed

disfluencies = set(['uh', 'um'])

def write_json(transcript, audiofile, offset, duration, output,
               nthreads=1, disfluency=False, conservative=False):
    """
    Force alignments of `audiofile` trimmed at (`offset`, `offset` + `duration`) with `transcript`. 
    The alignment is done over all the TED talks, since for the dev and test there is no discontinuities in the transcription.
    The results are saved in a json file inside a folder named after the TED speaker where the audio comes from (aka the name of the corresponding .sph / .wav file).
    
    Args:
    transcript: string containing the transcribed text
    audiofile: path to the .wav audiofile
    offset: float indicating where the speaker starts talking
    duration: float indicating the length of the talk (in seconds)
    output: path where results are saved
    nthreads: number of alignment threads (set to 1 if you are parallelizing this function)
    disfluency: bool, include disfluencies (uh, um) in alignment
    conservative: bool, conservative alignment 
    """
    print("Processing audio file ", audiofile)

    resources = gentle.Resources()

    with gentle.resampled(audiofile, offset=offset, duration=duration) as wavfile:
        print("Starting alignment")
        aligner = gentle.ForcedAligner(resources, transcript, nthreads=nthreads, disfluency=disfluency, 
                                       conservative=conservative, disfluencies=disfluencies)
        result = aligner.transcribe(wavfile)

    fh = open(output, 'w', encoding="utf-8") 
    fh.write(result.to_json(indent=2))
    fh.close()
    print("Output written to %s" % (output))

    
def get_transcript(path2transcript):
    """
    Compute the offset, duration and transcript of a given transcription file.
    
    Args:
    path2transcript: path to .stm transcription file
    
    Returns:
    offset: float indicating where the TED speaker start his talk.
    duration: length in seconds of the speech
    transcript: string containing the corresponding transcription
    """
    transcript = []
    offset = -1
    end = 0
    with open(path2transcript, 'rt') as f:
        records = f.readlines()
        for sentence in records:
            fields = sentence.split()
            label = fields[6:]
            if not 'ignore_time_segment_in_scoring' in label:
                if offset < 0:
                    offset = float(fields[3])
                transcript.extend(label)
                end = float(fields[4])
    transcript = " ".join(transcript)
    duration = end - offset
    return offset, duration, transcript    
    
    
def get_files(path):
    """
    Create 3 lists of same length;
    - the first contains paths audiofiles
    - the second contains the transcription of each of those audios
    - the third the offset and duration of the TED talk in the audios
    
    Note that 3 lists are aligned.

    Args:
    path: path to the folder containing the .wav audio files
    
    Returns:
    audiofiles: list of paths to the .wav audio file. 
    transcripts: list of the corresponding transcriptions .
    offsets_durs: list of tuples of corresponding offset and duration for th
    
    """    
    audiofiles = []
    transcripts = []
    offsets_durs = []
    
    for _, _, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                audiofiles.append(os.path.join(path, file))    
                path2txt = os.path.join(path, file).replace('wav', 'stm')
                
                offset, duration, transcript = get_transcript(path2transcript=path2txt)
                offsets_durs.append((offset, duration))
                transcripts.append(transcript)
                
    return audiofiles, transcripts, offsets_durs   


path2dev = '/aimlx/Datasets/TEDLIUM_release1/dev/wav'
path2test = '/aimlx/Datasets/TEDLIUM_release1/test/wav'

audiofiles, transcripts, offsets_durs = get_files(path=path2test)

Parallel(n_jobs=11, prefer="threads")(delayed(write_json)(transcript, audiofile, offset=off_dur[0], duration=off_dur[1], output=audiofile.replace('wav','json'))
                                        for transcript, audiofile, off_dur in zip(transcripts, audiofiles, offsets_durs))