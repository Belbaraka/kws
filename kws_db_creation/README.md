### Building a keyword spotting database using TED-LIUM

We provide in this folder scripts to build a kws database from TED talks. Keywords are extracted directly from the speeches which gives them context and hence could be used for an end2end approach for continuous speech keywords spotting. 

The **word-level** alignments (as well as phone alignments) are done using the opensource tool [gentle](https://github.com/lowerquality/gentle); a *robust yet lenient forced-aligner built on Kaldi*.

The keyword database is constructed as follows:
1. For each TED talk, we create a json file that contains the forced aligned words boundaries in the speech and other useful information (eg. whether the alignment was successful or not). Checkout `example.json` to know more about the file's structure.
2. From these json files we get the *positions* of the keyword in the audio file
3. Each occurence of the keyword in the audio file is extracted and saved as a `.wav` file.

[TODO : Tutorial of how to use the scripts to create the keyword spotting database]('')