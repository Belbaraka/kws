## ASR based keyword spotting system

We present a keyword spotting system solely based on a state of the art ASR engine. The pretrained model used can be found [here](http://zamia-speech.org/asr/). It has been trained on ~ **1500 hours** of speech (tedlium3, librispeech, voxforge and other open source datasets) and acheives **8.84% WER**. For more details check *Zamia*'s [release post.](https://goofy.zamia.org/lm/2019/06/20/1500-Hours-160k-Words-English-Zamia-Speech-Models-Released.html) The proposed KWS solution is straightforward; 
1. The user provides a wav file sampled at **16kHz**
2. The ASR engine performs the speech to text decoding
3. Finally the keywords are searched in the ASR's ouptut.

### How to
- First follow the steps in the notebook `baseline.ipynb` to prepare the keywords spotting module. 
- Upload the desired audio files in `input/audio`. Note that audio files must be sampled at **16kHz** and must be **less than 20 seconds long**.
- Create files `wav.scp` and `spk2utt`: `python3 prepare_files.py --path2audio='input/audio/' --path2files='transcriptions/' --path_in_docker='/opt/kaldi/egs/alibel_model/v1/input/audio/'`
- Launch docker image and attach repo to it: `docker run -it -v ~/kws/asr_kws/kaldi-generic-en-tdnn_f-r20190609:/opt/kaldi/egs/alibel_model kaldiasr/kaldi:latest`
- Run decoding script: `cd egs/alibel_model/v1` and `./decode.sh`. Transcriptions can be found in `transcriptions/transcribed_speech.txt`
- Run keyword spotting system: `python3 asr_kws.py --keywords 'keyword_1' 'keyword_2' --path2transcription='transcriptions/transcribed_speech.txt'`

