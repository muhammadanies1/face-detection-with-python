# FaceRecognitionUsing-PCA-2D-PCA-And-2D-Square-PCA
Implementation of PCA/2D-PCA/2D(Square)-PCA in Python for recognizing Faces:
1. Single Person Image
2. Group Image
3. Recognize Face In Video
# Accuracy on ORL dataset
  - PCA(93.42%)
  - 2D-PCA(96.05%)
  - 2D(Square)-PCA(97.36%)
# Requirements
1. numpy
2. opencv
3. scipy
# Usage
1. In Face_Recognition class use algo_type from (pca, 2d-pca, 2d2-pca)
2. In Face_Recognition class use reco_type as
  - for single image = 0
  - for video = 1
  - for group image = 2
3. The project uses ORL dataset, You can put your dataset in the images folder and change the name of the dataset in the dataset.py file
(You can create new dataset by extracting faces using FaceExtractor provided)
4. Install opencv library in your Environment and then Run Face_Recognition and Enjoy Project
