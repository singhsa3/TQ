# TQ
Codes
•	model_custom.py: It contains all the functions and model codes in one place
•	MFCC_Extract_all_features.ipynb : This notebook is used to extract MFCC , delta and delta-delta features 
•	w2v2_Extract_all_features.ipynb : This notebook is used to extract w2v2 layers
•	Datamaster.ipynb: It is used for data processing and creation of the input data, which is fed into the models
•	Loop_main.ipynb: It is used for running various models and dumping the results adb.log and in in results.txt

Database:
Final Result.csv contains results of all the experiments. This file was generated from the “consol” table from SQLite database adb.log 
Consol table data field descriptions:
Dataset: It indicates which type of dataset was used as inputs
Therapist: Only Sandra and Michelle were considered
Emotion: Only fear, anger, and sadness were considered
Modelname: Which model architecture was used 
Other fields are self explanatory , Epoch, Batch, train_loss, train_recall, train_binary_accuracy, train_precision, train_f1, val_los,val_recall, val_binary_accuracy, val_precsion, val_f1
