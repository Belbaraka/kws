"""
This class provides functions to extract features from audio files and generates data on the fly for the models to train on.
"""

import numpy as np
import pandas as pd
import glob
import csv
import librosa
import os
import subprocess
from helpers import *
#from python_speech_features import mfcc
from tqdm import tqdm


class DataGenarator():
    
    def __init__(self, path2data, path2features, n_mfcc=40):#, winlen=0.030, winstep=0.01):
        self.path2data = path2data
        self.path2features = path2features
        self.n_mfcc = n_mfcc
    
    def sph2wav(self, sph, wav):
        """Convert an sph file into wav format for further processing"""
        command = [
            'sox','-t','sph', sph, '-b','16','-t','wav', wav
        ]
        subprocess.check_call( command ) # Did you install sox (apt-get install sox)

    def process_tedelium(self, csv_file, category):

        parent_path = os.path.join(self.path2data, category)
        labels, wave_files, offsets, durs = [], [], [], []

        # create csv writer
        writer = csv.writer(csv_file, delimiter=',')

        # read STM file list
        stm_list = glob.glob(os.path.join(parent_path,'stm', '*'))
        
        for stm in tqdm(stm_list, desc='reading STM file list'):
            with open(stm, 'rt') as f:
                records = f.readlines()
                for record in records:
                    field = record.split()

                    # wave file name
                    wave_file = os.path.join(parent_path,'sph/%s.sph.wav' % field[0])
                    wave_files.append(wave_file)

                    # label index
                    labels.append(str2index(' '.join(field[6:])))

                    # start, end info
                    start, end = float(field[3]), float(field[4])
                    offsets.append(start)
                    durs.append(end - start)

        # save results
        for i, (wave_file, label, offset, dur) in tqdm(enumerate(zip(wave_files, labels, offsets, durs)), desc='saving results', total=len(wave_files)):
            fn = "%s-%.2f" % (wave_file.split('/')[-1], offset)
            path2feature = os.path.join(self.path2features, fn + '.npy')

            # load wave file
            if not os.path.exists( wave_file ):
                sph_file = wave_file.rsplit('.',1)[0]
                if os.path.exists( sph_file ):
                    sph2wav( sph_file, wave_file )
                else:
                    raise RuntimeError("Missing sph file from TedLium corpus at %s"%(sph_file))

            signal, sr = librosa.load(wave_file, mono=True, sr=None, offset=offset, duration=dur)

            # get mfcc feature
            mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=self.n_mfcc)

            # save result ( exclude small mfcc data to prevent ctc loss )
            if len(label) < mfcc.shape[1]:

                # save meta info
                writer.writerow([fn] + label)

                # save mfcc
                np.save(path2feature, mfcc, allow_pickle=False)    