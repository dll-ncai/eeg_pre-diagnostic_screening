# EEG anomaly detection 
  The files provided here classify the EEG records provided in the NMT dataset into normal and abnormal recordings. 
  This folder contains code that customizes CNN architectures (originally proposed by Schirrmeister et. al. [1]) for the NMT dataset. 
  ## Acknowledgement
  The code in this folder is based on the example provided by R. T. Schirrmeister for the brain decode library at the link below: 
  
   https://github.com/robintibor/auto-eeg-diagnosis-example

 The expeiments use NMT dataset for EEG classification using models based on Deep and Shallow CNN architechtures proposed by 
  R. T. Schirrmeister et al in `Deep  learning  with  convolutional  neural  networks  for  EEG  decodingand  visualization`
## Run
1. Download and unzip dataset in working directory
2. Install braindecode library available at  https://robintibor.github.io/braindecode/
3. Modify config.py, especially correct data folders for the dataset.
4. Run with `python ./auto_diagnosis.py`
5. Run with `python ./finetune.py` after adding path to model(pre-trained on TUH)
##
[1] R. Schirrmeister, L. Gemein, K. Eggensperger, F. Hutter and T. Ball, "Deep learning with convolutional neural networks for decoding and visualization of EEG pathology," 2017 IEEE Signal Processing in Medicine and Biology Symposium (SPMB), Philadelphia, PA, 2017, pp. 1-7, doi: 10.1109/SPMB.2017.8257015.
