'''
Creates files wav.scp and  spk2utt
- wav.scp : a 2 columns file, left utterance_id , right path to audio file inside docker image
- spk2utt : a 2 columns file, left speaker_id, right utterance_id
We assume that each audio file represents one utterance and that speaker_id is different across audio files.
    
Args:
path2audio: path to .wav audio files in local
path2files: path to the transcriptions folder 
path_in_docker: path to .wav audio files inside docker image (replace `alibel_model` with the name of the folder you created)

'''

from os import listdir
from os.path import isfile, join
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument('-p2a', '--path2audio', type=str, default='input/audio/',
                    help='path to .wav audio files in local')

parser.add_argument('-p2f', '--path2files', type=str, default='transcriptions/',
                    help='path to the transcriptions folder')

parser.add_argument('-p2d', '--path_in_docker', type=str, default='/opt/kaldi/egs/alibel_model/v1/input/audio/',
                    help='path to .wav audio files inside docker image')

args = parser.parse_args()

path2audio = args.path2audio
path2files = args.path2files
path_in_docker = args.path_in_docker

# list all audio files 
filenames = [f for f in listdir(path2audio) if isfile(join(path2audio, f)) and f.endswith('.wav')]

# prepare wav.scp file
with open(join(path2files, 'wav.scp'), mode='w+') as fp:
    for i, filename in enumerate(filenames):
        utt_id = filename.split('.wav')[0] + '_utt' + str(i)
        fp.write(utt_id + ' ' + join(path_in_docker, filename) + '\n')

print('File wav.scp created in ' + join(path2files, 'wav.scp'))        
        
# prepare spk2utt file
with open(join(path2files, 'spk2utt'), mode='w+') as fp:
    for i, filename in enumerate(filenames):
        utt_id = filename.split('.wav')[0] + '_utt' + str(i)
        spk_id = 'speaker_' + str(i)
        fp.write(spk_id + ' ' + utt_id + '\n')

print('File spk2utt created in ' + join(path2files, 'spk2utt'))   