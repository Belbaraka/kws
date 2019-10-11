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
from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences
from random import shuffle as shuf


class DataPrep():
    
    def __init__(self, path2data, path2features, pickle_filename, n_mfcc=26, n_mels=40, hop_length=160, frame_length=320):#, winlen=0.030, winstep=0.01):
        self.path2data = path2data
        self.path2features = path2features
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.pickle_filename = pickle_filename
    
    def sph2wav(self, sph, wav):
        """Convert an sph file into wav format for further processing"""
        command = [
            'sox','-t','sph', sph, '-b','16','-t','wav', wav
        ]
        subprocess.check_call( command ) # Did you install sox (apt-get install sox)

    def process_tedelium(self, category):

        parent_path = os.path.join(self.path2data, category)
        labels, wave_files, offsets, durs = [], [], [], []

        # create df to be filled with labels and ids
        df = pd.DataFrame(columns=['filename', 'frames', 'labels'])
        
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
            mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=self.n_mfcc, n_mels=self.n_mels, n_fft=self.frame_length, hop_length=self.hop_length)

            # save result ( exclude small mfcc data to prevent ctc loss )
            if len(label) < mfcc.shape[1]:

                # save meta info
                df = df.append({'filename': fn, 'frames' : mfcc.shape[1], 'labels' : label}, ignore_index=True)    
                    
                # save mfcc
                np.save(path2feature, mfcc, allow_pickle=False)    
        # save dataframe
        df.to_pickle(os.path.join(self.path2features, self.pickle_filename))
       
    
# DataGenerator following this tutorial: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
class DataGenerator(Sequence):
    """
    Thread safe data generator for the fit_generator
    
    Args:
        path2features (string): path to the mfcc features 
        pickle_filename (string): filename of the pickle file containing the dataframe (filename, frames, labels)
        batch_size (int): size of each batch
        mfcc_features (int, default=26): how many mfcc-features to extract for each frame
        epoch_length (int, default=0): the number of batches in each epoch, if set to zero it uses all available data
        shuffle (boolean, default=True): whether to shuffle the indexes in each batch
    """
    def __init__(self, path2features, pickle_filename, batch_size=32, mfcc_features=26, epoch_length=0, shuffle=True):
        
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.shuffle = shuffle
        self.path2features = path2features
        self.df = pd.read_pickle(os.path.join(path2features, pickle_filename))
        self.mfcc_features = mfcc_features
        
        # Initializing indexes
        self.indexes = np.arange(self.df.shape[0])

    def __len__(self):
        """Denotes the number of batches per epoch"""
        if (self.epoch_length == 0) | (self.epoch_length > int(np.floor(self.df.shape[0]/self.batch_size))):
            self.epoch_length = int(np.floor(self.df.shape[0] / self.batch_size))
        return self.epoch_length

    def __getitem__(self, batch_index):
        """
        Generates a batch of correctly shaped X and Y data
        :param batch_index: index of the batch to generate
        :return: input dictionary containing:
                'the_input':     np.ndarray[shape=(batch_size, max_seq_length, mfcc_features)]: input audio data
                'the_labels':    np.ndarray[shape=(batch_size, max_transcript_length)]: transcription data
                'input_length':  np.ndarray[shape=(batch_size, 1)]: length of each sequence (numb of frames) in x_data
                'label_length':  np.ndarray[shape=(batch_size, 1)]: length of each sequence (numb of letters) in y_data
                 output dictionary containing:
                'ctc':           np.ndarray[shape=(batch_size, 1)]: dummy data for dummy loss function
        """

        # Generate indexes of current batch
        indexes_in_batch = self.indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

        # Shuffle indexes within current batch if shuffle=true
        if self.shuffle:
            shuf(indexes_in_batch)

        # Preprocess and pad data
        x_data, input_length = self.load_and_pad_features(indexes_in_batch)
        y_data, label_length = self.load_and_pad_labels(indexes_in_batch)

        inputs = {'the_input': x_data,
                  'the_labels': y_data,
                  'input_length': input_length,
                  'label_length': label_length}

        outputs = {'ctc': np.zeros([self.batch_size])} # dummy data for dummy loss function

        return inputs, outputs

    def load_and_pad_features(self, indexes_in_batch):
        """
        Loads MFCC features
        Zero-pads each sequence to be equal length to the longest sequence.
        Stores the length of each feature-sequence before padding for the CTC
        :param indexes_in_batch: list of indices of data points in batch
        :return: x_data: numpy array with padded feature-sequence 
                 input_length: numpy array containing unpadded length of each feature-sequence
        """

        # Finds longest frame in batch for padding
        max_pad_length = self.df.loc[indexes_in_batch].frames.max()
        
        x_data = np.empty([0, max_pad_length, self.mfcc_features])
        len_x_seq = []

        # loading mfcc features and pad so every frame-sequence is equal max_x_length
        batch_size = len(indexes_in_batch)
        for idx, i in tqdm(zip(indexes_in_batch, range(0, batch_size)), desc='loading and padding mfcc features', total=batch_size):
            
            mfcc_frames = np.load( os.path.join(self.path2features, self.df.loc[idx].filename + '.npy') )
            mfcc_padded = pad_sequences(mfcc_frames, maxlen=max_pad_length, dtype='float', padding='post', truncating='post')
            mfcc_padded = mfcc_padded.T
            
            x_data = np.insert(x_data, i, mfcc_padded, axis=0)

            # Save number of frames before pading
            nb_frames = mfcc_frames.shape[1]
            len_x_seq.append(nb_frames - 2)  # -2 because ctc discards the first two outputs of the rnn network

        # Convert input length list to numpy array
        input_length = np.array(len_x_seq)
        return x_data, input_length

    def load_and_pad_labels(self, indexes_in_batch):
        """
        Load and pads labels (int sequences)
        :param indexes_in_batch: list of indices of data points in batch
        :return: y_data: numpy array with transcripts converted to a sequence of ints and zero-padded
                 label_length: numpy array with length of each sequence before padding
        """
        # Finds longest sequence in y for padding
        max_y_length = max(list(map(lambda x: len(x[1].labels), self.df.loc[indexes_in_batch].iterrows() ) ) )
        y_data = np.empty([0, max_y_length])
        len_y_seq = []

        # Converts to int and pads to be equal max_y_length
        batch_size = len(indexes_in_batch)
        for idx, i in tqdm(zip(indexes_in_batch, range(0, batch_size)), desc='loading and padding labels', total=batch_size):
            y_int = self.df.loc[idx].labels
            len_y_seq.append(len(y_int))

            for j in range(len(y_int), max_y_length):
                y_int.append(0)

            y_data = np.insert(y_data, i, y_int, axis=0)

        # Convert transcript length list to numpy array
        label_length = np.array(len_y_seq)

        return y_data, label_length