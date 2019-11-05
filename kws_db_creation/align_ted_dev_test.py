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
               nthreads=multiprocessing.cpu_count(), disfluency=False, conservative=False):
        
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
    
    
def get_files(path='/aimlx/Datasets/TEDLIUM_release1/dev/wav'):
    
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

audiofiles, transcripts, offsets_durs = get_files(path='/aimlx/Datasets/TEDLIUM_release1/dev/wav')


Parallel(n_jobs=11)(delayed(write_json)(transcript, audiofile, offset=off_dur[0], duration=off_dur[1], output=audiofile.replace('wav','json'))
                                        for transcript, audiofile, off_dur in zip(transcripts, audiofiles, offsets_durs))