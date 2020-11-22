# EEG classificaion
This code is released with NME and uses braindecode to test generalization of CNN on EEG anomaly detection as discussed in 
The NME Scalp EEG Dataset: An Open-Source Annotated Dataset of Healthy and Pathological EEG Recordings for Predictive Modeling
## Requirements
1. Depends on https://robintibor.github.io/braindecode/ 
2. This code was programmed in Python 3.6 (might work for other versions also).

## Run
1. Modify config.py, especially correct data folders for your path..
2. Run with `python ./auto_diagnosis.py`
##
## Acknowledgment 
The shallow and Deep CNN experiments are built on example provided for BrainDecode by ‪Robin Tibor Schirrmeister here
https://github.com/robintibor/auto-eeg-diagnosis-example

Also chrononet experiments are based on implementation from the following package:
Kunal Patel et al: https://github.com/kunalpatel1793/Neural-Nets-Final-Project
