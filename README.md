# The NMT Scalp EEG Dataset: An Open-Source Annotated Dataset of Healthy and Pathological EEG Recordings for Predictive Modeling

* [Introduction](#introduction)
* [Requirements](#requirements)
* [Dataset](#dataset)
* [Citation](#citation)
 # Introduction
This repo contains the source code for 'Pre-Diagnostic Screening of Abnormal EEG' using **"NMT Scalp EEG Dataset"**. The study compares the performance of state-of-the-art deep learning algorithms on the task of EEG abnormality classification on the NMT and Temple Univeristy Hospital (TUH) Abnormal EEG dataset. The details of our research on pre-diagnostic anomaly detection can be found in the following paper:

* Hassan Aqeel Khan, Rahat Ul Ain, Awais Mehmood Kamboh, Hammad Tanveer Butt, Saima Shafait, Wasim Alamgir, Didier Stricker and Faisal Shafait (2022) **"The NMT Scalp EEG Dataset: An Open-Source Annotated Dataset of Healthy and Pathological EEG Recordings for Predictive Modeling",** Frontiers in Neuroscience. doi: 10.3389/fnins.2021.755817

# Requirements
1. Install braindecode library available at https://robintibor.github.io/braindecode/ 
2. This code was programmed in Python 3.6 (might work for other versions also).
3. Dataset available at: https://dll.seecs.nust.edu.pk/downloads/

# Dataset
The NMT dataset consists of 2,417 EEG records at this time. The EEG montage used for recordings consists of the standard 10-20 system and is shown in Figure below.
<br/>
<img src="Linked-ear-referenced-standard-electrode-montage_W640.jpg" alt="Linked-ear-referenced-standard-electrode-montage" width="400"/>
<br/>
There are 19 channels on the scalp, channels A1 and A2 are reference channels on auricle of the ear. The sampling rate of all channels is 200 Hz. The average duration of each record is 15 min. The histogram of recording lengths is given in Figure below.
<br/>
<img src="The-number-of-recordings-in-the-NMT-dataset-for-each-range-of-duration-in-minutes_W640.jpg" alt="The-number-of-recordings-in-the-NMT-dataset-for-each-range-of-duration-in-minutes" width="400"/>
<br/>
The histograms of age distributions of males and female subjects in the dataset are shown in Figure below.
<br/>
<img src="Histogram-of-age-distribution-in-the-NMT-dataset-The-shaded-regions-indicate-the_W640.jpg" alt="Histogram-of-age-distribution-in-the-NMT-dataset" width="400"/>
<br/>
The age ranges from under 1 year old up to 90 years old; 66.56 and 33.44% of the records are collected from male and female subjects, respectively. 16.17% of EEG recordings from males are abnormal/pathological whereas, in case of females, 19.18% records are abnormal/pathological.



 # Citation
This repo was used to generate the results for the following paper on Pre-Diagnostic Screening of Abnormal EEG.
  
  > Citation: Khan HA, Ul Ain R, Kamboh AM, Butt HT, Shafait S, Alamgir W, Stricker D and Shafait F (2022) **The NMT Scalp EEG Dataset: An Open-Source Annotated Dataset of Healthy and Pathological EEG Recordings for Predictive Modeling.** Front. Neurosci. 15:755817. doi: 10.3389/fnins.2021.755817

**BibTex Reference:**
```console
@ARTICLE{10.3389/fnins.2021.755817,
  
author={Khan, Hassan Aqeel and Ul Ain, Rahat and Kamboh, Awais Mehmood and Butt, Hammad Tanveer and Shafait, Saima and Alamgir, Wasim and Stricker, Didier and Shafait, Faisal},   
	 
title={The NMT Scalp EEG Dataset: An Open-Source Annotated Dataset of Healthy and Pathological EEG Recordings for Predictive Modeling},      
	
journal={Frontiers in Neuroscience},      
	
volume={15},      
	
year={2022},      
	  
url={https://www.frontiersin.org/article/10.3389/fnins.2021.755817},       
	
doi={10.3389/fnins.2021.755817},      
	
issn={1662-453X},

Keywords = { open-source EEG dataset, automated EEG analytics, pre-diagnostic EEG screening, computer aided diagnosis, computational neurology, convolutional neural networks, deep learning, generalization performance}
}
```
