#!/bin/bash

# Inspired by http://jrmeyer.github.io/asr/2017/01/10/Using-built-DNN-model-Kaldi.html

. ./path.sh

### DECODING ###

# MFCC FEATURES
compute-mfcc-feats \
    --config=conf/mfcc_hires.conf \
    scp:transcriptions/wav.scp \
    ark,scp:transcriptions/feats.ark,transcriptions/feats.scp;

# NORMALIZED FEATURE VECTORS 
# This is optional, and only really makes sense if you have a lot of 
# recordings with repeat speakers
# compute-cmvn-stats \
#     --spk2utt=ark:transcriptions/spk2utt \
#     scp:transcriptions/feats.scp \
#     ark,scp:transcriptions/cmvn.ark,transcriptions/cmvn.scp;

# apply-cmvn \
#     --norm-means=false \
#     --norm-vars=false \
#     --utt2spk=ark:transcriptions/utt2spk \
#     scp:transcriptions/cmvn.scp \
#     scp:transcriptions/feats.scp \
#     ark,scp:transcriptions/new-feats.ark,transcriptions/new-feats.scp;

# IVECTOR FEATURES
ivector-extract-online2 \
	--config=ivectors_test_hires/conf/ivector_extractor.conf \
	ark:transcriptions/spk2utt \
	scp:transcriptions/feats.scp \
	ark,scp:transcriptions/ivectors.1.ark,transcriptions/ivectors.1.scp;

# TRAINED DNN-HMM + FEATURE VECTORS --> LATTICE
nnet3-latgen-faster \
	--online-ivectors=scp:transcriptions/ivectors.1.scp --online-ivector-period=10 \
	--frame-subsampling-factor=3 --frames-per-chunk=140 --extra-left-context=0 --extra-right-context=0 \
	--extra-left-context-initial=-1 --extra-right-context-final=-1 --minimize=false --max-active=7000 --min-active=200 --beam=15.0 --lattice-beam=8.0 \
	--acoustic-scale=1.0 --allow-partial=true \
    --word-symbol-table=model/graph/words.txt \
    model/final.mdl \
    model/graph/HCLG.fst \
    ark:transcriptions/feats.ark \
    ark,t:transcriptions/lattices.ark;

echo 'Success nnet3-latgen-faster'

# LATTICE --> BEST PATH THROUGH LATTICE
lattice-best-path \
    --word-symbol-table=model/graph/words.txt \
    ark:transcriptions/lattices.ark \
    ark,t:transcriptions/transcribed_integers.tra;

# BEST PATH INTERGERS --> BEST PATH WORDS
int2sym.pl -f 2- \
    model/graph/words.txt \
    transcriptions/transcribed_integers.tra \
    > transcriptions/transcribed_speech.txt;


