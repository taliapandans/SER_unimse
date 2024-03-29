# Speech Emotion Recognition
This is a university project to build a speech emotion recognition with multiple modalities.
In this case we sample UniMSE paper: 
- https://github.com/LeMei/UniMSE
- https://arxiv.org/pdf/2211.11256.pdf

### Datasets
- MOSEI
- MOSI
- IEMOCAP

The dataset folders are to be found under UniMSE>Simcse>dataset [folder](https://drive.google.com/drive/folders/1CFy5kgNZDS8FEcDtL_57dsQBNRpSFhHf?usp=drive_link) 
### Installation and Environment
Unimse_Submission.ipynb contains a walkthrough of all the installation steps. In the end of the notebook, main.py is tested for three different datasets (MOSEI, MOSI, IEMOCAP)

### Process
- First, every single dataset (MOSI, MOSEI, IEMOCAP，MELD)  is preprocessed by running `preprocess.py`. For every dataset, a train, test and validation pkl is created and saved to the according folders.<br /> 
- Under UniMSE>src, the paths to the created .pkl files must be set correctly for every dataset. The train dataset and number of epochs can be changed on the `config.py`. <br /> 
- Please also add [t5-base](https://drive.google.com/drive/folders/1T3jRd_AMwdmAz5lRFW_aliokklaWW3ju?usp=drive_link) to `./src` folder. 
- Lastly, model is trained by running `main.py`.

### Current results (will be updated soon)
MOSEI: | Train Loss: 0.0070 | Valid Loss 0.3053 | Test Loss 0.3024 <br/> 
MOSI:  Train Loss: 0.0121 | Valid Loss 0.4356 | Test Loss 0.3878

### Current Goals
Run on IEMOCAP <br/> 
Fine Tune on Emodb


