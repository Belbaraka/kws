## Folder's structure
We propose in what follows scripts used to detect a given set of keywords in continuous speech.

**Scripts**

- __`preprocessing.py`__ : contains functions used to perform pre-processing (before training).

- __`losses.py`__ : script which contains custom loss and scoring functions (*focal loss*, *f1-score*...).

- __`models.py`__ : different models all CNN based (mainly *res-nets* and *deep CNN*).

- __`postprocessing.py`__ : contains functions used to perform post-processing (after training).

- __`scoring.py`__ : contains functions that cleans test data (removing music part, clapping, ..) and computes fom results 

- __`train.py`__ : script which trains a model to spot arbitrary list of keywords. 
    - *eg*: `python3 train.py -kw 'people' 'world' --model 'dnn' --path2kwdb '~/TEDLIUM_release-3/data/1000_kws_db' -out '~/kws/e2e_kws/'`
    

- __`evaluate.py`__ : script that evaluates the models and compute FOM results on test ted talks.
    - *eg*: `python3 evaluate.py -kw 'people' 'world'  -p2m  '~/models/set1_models/dnn_set1_5_epochs.h5' -p2w '~/TEDLIUM_release-3/data/wav' -p2j '~/TEDLIUM_release-3/data/aligned_json' -p2ted '~/test_data/test_ted_talks_50.npy'`


 
 


**Folders**

- *notebooks*
    - __`db_statitics_ted_1.ipynb`__ : statistics about the keyword database extracted from TEDLIUM's first release

    - __`db_statitics_ted_3.ipynb`__ : statistics about the keyword database extracted from TEDLIUM's third release

    - __`e2e_kws_tedlium.ipynb`__ : notebook that performs the E2E-CSKWS all the way through (*pre-processing*, *training*, *postprocessing*, *testing* and *plotting*)
    
- *demo*
    - __`demo.ipynb`__ : demo notebook, extract keyword occurrences given a test speech.
    
    - __`demo.py`__ : script which contains functions used for the demo. 
    
    - __`models/`__ : contains trained models (*.h5* file)
        - __set1_models/__ : models trained to detect keywords : *people*, *because*, *think*, *world*
        - __set2_models/__ : models trained to detect keywords : *something*, *different*, *actually*, *important*
        - __set3_models/__ : models trained to detect keywords : *another*, *percent*, *problem*, *technology*
        - __set4_models/__ : models trained to detect keywords : *years*, *little*, *through*, *together*
    





