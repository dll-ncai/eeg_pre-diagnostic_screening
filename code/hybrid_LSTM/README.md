# Hybrid CNN-LSTM Model for EEG classification
A  hybrid  model  is  designed  to  solve  EEG  yield  problem using  features  learned  from  Deep  CNN discussed here
`The NMT Scalp EEG Dataset: An Open-Source Annotated Dataset of Healthy and Pathological EEG Recordings for Predictive Modeling`
## Run
1. Modify `config.py`, especially correct data folders for your dataset.
2. `diagnose.py` loads  `shallow/deep.pt` model to extract features from EEG signals and store in .mat files
3. `hybrid_LSTM.py` loads sequence of features from .mat files to train LSTM model for classification
##
