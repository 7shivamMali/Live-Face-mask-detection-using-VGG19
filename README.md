# Face-Mask-Detection-using-VGG-19

by,<br>
Shivam Mali<br>
Abhishek Pandey<br>


This Repository contains source code and dataset of our project "Face Mask Detection using VGG 19"

### Dataset 

The Face Mask detection using Machine Learning uses the facemask_Detection Dataset which is available on Github.<br>
dataset link - https://github.com/techyhoney/Facemask_Detection/tree/master/dataset<br>
The facemask_detection Dataset includes 4000 images in total (2000 for each with_mask and without_mask).<br>
dataset structure - <br>
dataset - 
* with_mask
* without_mask<br>

### Source code 

In this project for the face mask detection task we have used a VGG-16 model which reached training accuracy of 99% and testing accuracy of 99%.
The requirements.txt file contain list of all required python libraries for each python file 
The code has three files - 
* main1.ipynb - code to train the model
* live_FMD.py - code for live face mask detection
* FMD.py - code for face mask detection on input image

The live_FMD.py and FMD.py uses the trained model (FMD_VGG19_D1.h5) so if anyone want to try the trained model they can simply download the repo install the required libraries and run the live_FMD.py and FMD.py for live face mask detection and face mask detection on image respectively

