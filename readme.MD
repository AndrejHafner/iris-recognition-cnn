# Iris recognition with CNN for feature extraction

### TODO:
- cite the authors of the original pipeline
- try normalizing pictures with the mean and std of the CASIA DB, not imagenet params


### Casia-Iris-Interval structure
The images of CASIA-Iris-Interval are stored as:  
$root path$/CASIA-Iris-Interval/YYY/E/S1YYYENN.jpg  
YYY: the unique identifier of the subject in the subset  
E: ‘L’ denotes left eye and ‘R’ denotes right eye  
NN: the index of the image in the class   


### Casia-Iris-Thousand
- 25 non-normalized in train images and 9 in test - with enhancement


### ResNet50d
- 86% acc - first 5 layers + global pool + no mask

### EfficientNet B3a
- 87% acc - first 4 layers + no global pool + no mask

### Credits
https://github.com/Linchunhui/Iris_Recognition  
https://github.com/ngoclamvt123/usit-v2.2.0

