Script.py => params optimization
idm_output.txt => all frames with optimized (calibrated) parameters
idm_mean_parameters.csv => since in 95% algorithms have to the same values, here are the values for future clustering.
driver_classification_summary.txt => simple classification output
driver_classification.png => plot 
08.05 => presentation


1.) RUN THIS TO INSTALL ALL IMPORTS
    pip install pandas numpy scipy scikit-learn matplotlib

2.) SEE DATA SECTION TO CHANGE DATASET
    META_DATA_PATH = 'tracks_meta'
    RAW_DATA_PATH = '241202_highD_lcTrainingTestForOptimisationData_di.pkl' <- please add it to folder
    MAX_REC_ID = 10 # [1-60]

3.) RUN FILE TO RUN THE PROGRAM
    python script.py

classification.py <= run after
graph.py <= run to show difference VIdm and Xacc
