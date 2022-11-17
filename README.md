# Human_Activity_Recognition
Human Activity Recognition (HAR) is a problem that is an active research field in pervasive computing. An HAR system has the main goal of analyzing human activities by observing and interpreting ongoing events.

Human Activities and Postural Transitions Data set (HAPT) data set is publicly available at: https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions.

The data set contains data from tri-axial accelerometer and gyroscope of a smart-phone, both captured at a frequency of 50 Hz. The data set consists of six basic activities (static and dynamic) and six postural transitions between the static activities. The static activities include standing, sitting and lying. The dynamic activities include walking, walking downstairs and walking upstairs. stand-to-sit, sit-to-stand, sit-to-lie, lie-to-sit, stand-to-lie, and lie-to-stand are the classes for postural transitions between the static activities.

The dataset contains:
1. Raw signal data of all the experiments from all participant and the labels of the per-formed activities.

2. Records of activity windows composed of feature vectors with time and frequency domain variables and their labels.

We will primarily use the feature vectors in this project, with the use of raw signal data with associated label can be used as an auxiliary experiment. Data set has to
be split further into train, validation, and test datasets based on the subjects. Use the following splits for the datasets from the raw data set (the train/validation/test split is approximately 70%/10%/20%):
1. Train dataset: user-01 to user-21.

2. Validation dataset: user-28 to user-30

3. Test dataset: user-22 to user-27

## Additional Hints:
1. The units used for the accelerations (total and body) are ’g’s (gravity of earth which is 9.80665 m/s2)

2. The gyroscope units are rad/sec.


## Acknowledgements  
Jorge L. Reyes-Ortiz(1,2), Davide Anguita(1), Luca Oneto(1) and Xavier Parra(2) 
1 - Smartlab, DIBRIS - Università  degli Studi di Genova, Genoa (16145), Italy. 
2 - CETpD - Universitat Politècnica de Catalunya. Vilanova i la Geltrú (08800), Spain   
har '@' smartlab.ws 
www.smartlab.ws

Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.
UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/index.php
Youtube video, Activity Recognition Experiment Using Smartphone Sensor, https://www.youtube.com/watch?v=XOEN9W05_4A, Oct 19, 2012
Udacity - for setting up the project and course contents.
