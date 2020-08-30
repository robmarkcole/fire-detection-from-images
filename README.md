# fire-detection-from-images
Detect fire in images using neural nets.

The purpose of this repo is to first identify relevant datasets and publications, then demonstrate a state of the art fire detection model that can be deployed to an edge device. Explore speed/accuracy tradeoffs & determine practical solution to deployment. Biggest challenge appears to be gathering comprehensive yet representative training dataset.

## Tooling and approach
* Frames will be fed through a fire/smoke/normal classifier. On positive detecion of fire/smoke frames are fed to an object detection model to determine the size/severity of the fire.
* Remain open to pytorch & tensorflow2
* Use Google Colab for training and host images on Google drive which has a nice UI, desktop apps with sync, easy auth
* Additionally use kaggle for training and data hosting? Not personally a fan of the kaggle UI, but keeping everything on kaggle simplifies auth, integration, version control, reproducibility
* Identify any relevant guidance/legislation on required accuracy of fire detection techniques

## Articles & repos
* [Fire and smoke detection with Keras and Deep Learning by pyimagesearch](https://www.pyimagesearch.com/2019/11/18/fire-and-smoke-detection-with-keras-and-deep-learning/) - dataset collected by scraping Google images (provides link to dataset with  1315 fire images), binary Fire/Non-fire classification with tf2 & keras sequential CNN, achieve 92% accuracy, concludes that better datasets are required
* [Fire Detection from scratch using YOLOv3](https://medium.com/@b117020/fire-detection-using-neural-networks-4d52c5cd55c5) - discusses annotation using LabelImg, using Google drive and Colab, deployment via Heroku and viz using Streamlit
* [fire-detect-yolov4](https://github.com/gengyanlei/fire-detect-yolov4) - training of Yolo v4 model
* [midasklr/FireSmokeDetectionByEfficientNet](https://github.com/midasklr/FireSmokeDetectionByEfficientNet) - Fire and smoke classification and detection using efficientnet, pytorch, visualizes the activation map
* [arpit-jadon/FireNet-LightWeight-Network-for-Fire-Detection](https://github.com/arpit-jadon/FireNet-LightWeight-Network-for-Fire-Detection) - A Specialized Lightweight Fire & Smoke Detection Model for Real-Time IoT Applications (e.g. on RPi), accuracy approx. 95%. Paper https://arxiv.org/abs/1905.11922v2
* [tobybreckon/fire-detection-cnn](https://github.com/tobybreckon/fire-detection-cnn) - links to a couple of datasets
* [EmergencyNet](https://github.com/ckyrkou/EmergencyNet) - identify fire and other emergencies from a drone
* [Fire Detection using CCTV images — Monk Library Application](https://medium.com/towards-artificial-intelligence/fire-detection-using-cctv-images-monk-library-application-242df1fca2b9) - keras classifier on kaggle datasets, mobilenet-v2, densenet121 and densenet201

## Datasets
* [FireNET](https://github.com/OlafenwaMoses/FireNET) - approx. 500 images with bounding boxes
* [cair/Fire-Detection-Image-Dataset](https://github.com/cair/Fire-Detection-Image-Dataset) - This dataset contains normal images and images with fire, and is highly unbalanced to reciprocate real world situations.
* [mivia Fire Detection Dataset](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/) - approx. 30 videos
* [USTC smoke detection](http://smoke.ustc.edu.cn/datasets.htm) - links to various sources that provide videos of smoke
* fire/not-fire dataset in the pyimagesearch article. Note that there are many umages of fire scenes that do not contain actual fire, but burnt out homes for example 
* [Fire Detection from CCTV on Kaggle](https://www.kaggle.com/ritupande/fire-detection-from-cctv) - images and video
* [FIRE Dataset on Kaggle]() - 755 outdoor-fire images and 244 non-fire images 

## Comments
* firenet is a VERY common name for model, do not use
