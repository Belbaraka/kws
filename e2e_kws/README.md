<h2><center>End-2-End continuous speech keyword spotting (E2E-CSKWS)</center></h2>

![](e2e-cskws.png)

### Folder's structure
We propose in what follows scripts used to detect a given set of keywords in continuous speech. For more details, please refer to the [wiki page](https://gitlab.com/SpeechMasterStudents/kws/-/wikis/home)

**Scripts**

- __`evaluate.py`__ : script that evaluates the models and compute FOM results on test ted talks.
 
- __`losses.py`__ : script which contains custom loss and scoring functions (*focal loss*, *f1-score*...).
 
- __`models.py`__ : different models all CNN based (mainly *res-nets* and *deep CNN*).
 
- __`postprocessing.py`__ : contains functions used to perform post-processing (after training).
 
- __`preprocessing.py`__ : contains functions used to perform pre-processing (before training).
  
- __`train.py`__ : script which trains a model to spot arbitrary list of keywords. 

- __`scoring.py`__ : contrains functions that cleans test data (removing music part, clapping, ..) and computes fom results 

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
    





