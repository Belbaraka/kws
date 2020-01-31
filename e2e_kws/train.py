import os
import argparse
import numpy as np
from random import sample
from tqdm import tqdm
import scipy.io.wavfile as wav
import warnings
warnings.filterwarnings('ignore')
import keras
from keras.utils import to_categorical
from models import *
from preprocessing import generate_sets, compute_mfcc, get_X_y


# Instantiate the parser
parser = argparse.ArgumentParser()
    
parser.add_argument('-kw', '--keywords', type=str, nargs='+',
                    help='customizable list of keywords to spot', required=True)

parser.add_argument('-m', '--model', type=str, default='dnn',
                    help='model to train (`baseline`, `dnn`, `parada`)')
    
parser.add_argument('-epo', '--epochs', type=int, default=4, 
                    help='number of epochs')

parser.add_argument('-p2kwdb', '--path2kwdb', type=str, default='/aimlx/Datasets/TEDLIUM_release-3/data/1000_kws_db',
                    help='path to the keyword database')    

parser.add_argument('-out', '--outputpath', type=str, default='/aimlx/kws/e2e_kws/',
                    help='path where the trained model is saved')
    
args = parser.parse_args()

keywords = args.keywords
_model = args.model
_epochs = args.epochs
_path2kw_db = args.path2kwdb
_outputpath = args.outputpath

#kws_sets = [['people', 'because', 'think', 'world'], ['something', 'different', 'actually','important'], 
#            ['another', 'percent', 'problem', 'technology'], ['years', 'little', 'through', 'together']]

#####################################
# Prepare data and compute features #
#####################################

print("[1/5] : Preparing keywords and non keywords data")

words_1000 = [] 
with open('1000-midlong', 'r') as thousend_words:
    for word in thousend_words:
        words_1000.append(word.strip())


# Prepare keywords and non keywords data
possible_non_keywords = list(set(words_1000) - set(keywords))
non_keywords = sample(possible_non_keywords, k=600)


# Test TED talks on which we will test our models
test_ted_talks = np.load('test_data/test_ted_talks_50.npy', allow_pickle=True)


# Load .wav files from keywrd database
path = os.path.abspath(_path2kw_db)
filenames = []

print("[2/5] : Loading .wav files from keyword database")

for w in tqdm(keywords + non_keywords):
    current_path = os.path.join(path, w[0], w) 
    for _, _, files in os.walk(current_path):
        for file in files:
            if all(test_ted_talk not in file for test_ted_talk in test_ted_talks):
                filenames.append(os.path.join(current_path, file))
                
    
print("[3/5] : Spliting and computing MFCC features for training and validation sets")
  
training, validation, testing = generate_sets(filenames, keywords, frame_size=1.0, n_mfcc=40, validation_percentage=15, 
                                              testing_percentage=0, add_noise=True, manipulate_pitch=True, add_volume=True)

# Get dimension along the x-axis of the MFCC frame
fs, sig = wav.read(filenames[0])
xdim, num_features = compute_mfcc(sig, fs, threshold=1.0, is_kw=True, num_features=40, add_noise=False)[0].shape


# Shape features in correct format or models to train on
X_train, y_train = get_X_y(training, xdim=xdim, num_features=num_features)
X_validation, y_validation = get_X_y(validation, xdim=xdim, num_features=num_features)
y_train, y_validation = to_categorical(y_train), to_categorical(y_validation)


##################
# Training model #
##################

if _model == 'baseline':
    model = baseline_dnn(len(keywords), xdim=xdim, num_features=num_features)
elif _model == 'parada':
    model = cnn_parada(len(keywords), xdim=xdim, num_features=num_features)
else:
    model = dnn_model(len(keywords), xdim=xdim, num_features=num_features)

print('[4/5] : Training model')

model.fit(X_train, y_train, batch_size=200, epochs=_epochs, verbose=1, validation_data=(X_validation, y_validation))

# Naming convention is model type, set on which it is trained, number of epochs it trained on
model_name = _model + '_' + str(_epochs) + '_'  'epochs.h5'

print('[5/5] : Saving model')
model.save(os.path.join(_outputpath, model_name ))

print('Trainig finished! Model ' + model_name + ' saved in ' + _outputpath)