  **The NMT EEG Scalp EEG Dataset**
====================================================================================================
    


The EEG records are available in the open-source EDF format. Below is a description of the 
of the files and the directory structure of this dataset.

1. `./Labels.csv`: This file contains a list of the EEG records along with demographic information 
and the ground-truth label. Below is a brief description of each column in this file.
	
	a. `recordname`: This column contains the name of each recordname
	
	b. `label`: The ground-truth label assigned to the record by the team of neurologists. This 
	          column contains one of two labels: 'normal' and 'abnormal' indicating normal and 
			  pathological EEG recording.
			  
	c. `age`: Age of the patient (in years).
	
	d. `gender`: Gender of the patient. This columns contains one of three labels: `'male'`, `'female'`
			   or 'not specified'.
	
	e. `loc`: Location of the file, indicating whether this record is included in the 'training' or
			'evaluation' (or test) set. For example, record 0000024.edf has label = normal and 
			loc = eval; this means that this record can found in the './normal/eval' directory. 
			Similarly, the record 0000025.edf has label = abnormal and loc = train; this means
			that this record can found in the './abnormal/train' directory.
			
2. `./DataStat.py`: This is a python script which can be used to plot the population Pyramid of the 
demographic information given in the 'Labels.csv' file and save it as a high-resolution .png 
image. This file can also be employed to examine different statistics about the dataset. Our
plan is to continune to add records to the NMT dataset and this script maybe useful for obtaining
statisitics of future iterations of the dataset.

3. `./abnormal`: This directory contains all EEG records labelled as 'abnormal' by the team of neurologists.
Files within in this directory are organized under two sub-directories 
	1. `'./abnormal/train'` which 
contains all abnormal EEG records that were used for training purposes in all our experiments and
	2. `'abnormal/eval'` which contains all abnormal EEG records that were used for evaluating/testing
performance of all algorithms discussed in our paper.

4. `./normal`: This directory contains all EEG records labelled as 'normal' by the team of neurologists.
Files within in this directory are organized under two sub-directories 
	1. `'./normal/train'` which 
contains all normal EEG records that were used for training purposes in all our experiments and
	2. `'normal/eval'` which contains all normal EEG records that were used for evaluating/testing
performance of all algorithms discussed in our paper.


===============================================================================
<br/>
Code Available at: https://github.com/dll-ncai/eeg_pre-diagnostic_screening
<br />
Dataset available at: https://dll.seecs.nust.edu.pk/downloads/
<br />
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

<br />
Last Updated: 01-01-2022	
