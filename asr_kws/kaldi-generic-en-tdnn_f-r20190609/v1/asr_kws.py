'''
ASR based keyword spotting system. This script searches for the keywords given in the output of the ASR.

Args:
keywords: customizable list of keywords.
path2transcription: path to the transcription file (ie. path to file `transcribed_speech.txt`)

Output:
Dict, {keyword: number_of_times_spotted}
'''

import argparse

# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument('-kw', '--keywords', type=str, nargs='+',
                    help='customizable list of keywords', required=True)

parser.add_argument('-p2t', '--path2transcription', type=str, default='transcriptions/transcribed_speech.txt',
                    help='path to the transcription file (ie. path to file `transcribed_speech.txt`)')

args = parser.parse_args()

keywords = args.keywords
path2transcription = args.path2transcription

kw_dict = {}
# Initialize keyword dictionary
for kw in keywords:
    kw_dict[kw] = 0
    
# Dict with elements (utterance_id, transcription)
utterances_dict = {} 

with open(path2transcription) as fp:
    for line in fp:
        utt_id, transcription = line.split(' ', 1)
        for kw in keywords:
            if kw in transcription:
                kw_dict[kw] += 1
        utterances_dict[utt_id] = transcription

print('\n**********************************************************************\n')        
print('ASR based keyword spotting results (keyword: number_of_times_spotted):')
print(kw_dict)
print('\n**********************************************************************')   
