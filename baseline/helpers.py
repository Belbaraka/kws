import numpy as np
import csv
import string


#
# vocabulary table
#

# index to byte mapping
alphabet = ['<EMP>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
              'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# byte to index mapping
char2index = {}
for i, ch in enumerate(alphabet):
    char2index[ch] = i

# vocabulary size
voca_size = len(alphabet)


# convert sentence to index list
def str2index(str_):

    # clean white space
    str_ = ' '.join(str_.split())
    # remove punctuation and make lower case
    str_ = str_.translate(str.maketrans('','',string.punctuation)).lower()
    
    res = []
    for ch in str_:
        try:
            res.append(char2index[ch])
        except KeyError:
            # drop Out Of Vocabulary
            pass
    return res


# convert index list to string
def index2str(index_list):
    # transform label index to character
    str_ = ''
    for i in index_list:
        if i > 0:
            str_ += alphabet[i]
        elif i == 0:  # <EOS>
            break
    return str_


# print list of index list
def print_index(indices):
    for index_list in indices:
        print(index2str(index_list))


# real-time wave to mfcc conversion function
def _load_mfcc(src_list):

    # label, wave_file
    label, mfcc_file = src_list

    # decode string to integer
    label = np.fromstring(label, np.int)

    # load mfcc
    mfcc = np.load(mfcc_file, allow_pickle=False)

    # speed perturbation augmenting
    mfcc = _augment_speech(mfcc)

    return label, mfcc


def _augment_speech(mfcc):

    # random frequency shift ( == speed perturbation effect on MFCC )
    r = np.random.randint(-2, 2)

    # shifting mfcc
    mfcc = np.roll(mfcc, r, axis=0)

    # zero padding
    if r > 0:
        mfcc[:r, :] = 0
    elif r < 0:
        mfcc[r:, :] = 0

    return mfcc