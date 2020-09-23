# fire-detection-from-images
The purpose of this repo is to demonstrate a fire detection neural net that can be deployed to an edge device, presumed to be connected to a camera that might be in a fixed location or on a roving drone.

## Motivation and challenges
Traditional smoke detectors work by [detecting the physical presence of smoke particles](https://www.nfpa.org/Public-Education/Staying-safe/Safety-equipment/Smoke-alarms/Ionization-vs-photoelectric). However they are prone to false detections (e.g. from toasters) and do not localise the fire particularly well. In these situations a camera solution could complement a traditional detector, in order to improve response times or to provide additional metrics such as the size and location of a fire. With the location and nature of the fire identified, an automated intervention may be possible, e.g. via a sprinkler system or drone. Also data can be sent to fire services to provide otherwise non-existent situational awareness. Particular locations I am interested in are: kitchens & living rooms, garages and outbuildings, and areas where fires might already be present but spreading outside a desired zone e.g. fire pit.

There are a couple of significant challenges & open questions:
* Gathering or locating a comprehensive, representative and balanced training dataset
* Handling different viewpoints, different camera manufacturers and settings, and different ambient lighting conditions.
* Since fires are so bright they can often wash out images and cause other optical disturbances, or use [water based](https://www.hackster.io/ben-eagan/fire-from-water-9e6ae4) fire simulators
* Is a single 'super' model preferable, or many more specialised models?
* Since we expect the model will have limitations, how do we make the model results interpretable?

Ideas:
* Preprocessing images, e.g. to remove background or apply filters
* Classifying short sequences of video, since the movement of fire is quite characteristic
* Simulated data, identify any software which can generate realistic fires and add to existing datasets
* Augmentations to simulate effect of different cameras and exposure settings
* Combining multiple signals including [thermal](http://www.nationalfirefighter.com/blog/Technological-Advances-A-Closer-Look-at-Thermal-Imaging) & priors to improve [ROC characteristics](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

## Tooling and approach
* Frames will be fed through neural net. On positive detection of fire metrics are extracted. Ignore smoke for MVP. Try various architectures & parameters to establish a 'good' baseline model.
* Remain open to pytorch & tensorflow2. However for low cost solution probably tensorflow-lite should be used.
* Use Google Colab for training and host images on Google drive which has a nice UI, desktop apps with sync, easy auth. Additionally or alternatively use kaggle, any particular advantages vs colab?
* Identify any relevant guidance/legislation on required accuracy of fire detection techniques
* [Seek thermal camera addon](https://www.amazon.co.uk/Seek-Thermal-UT-EAA-Protective-Waterproof-Black/dp/B00Y2QO6N0?th=1) is relatively low cost and can be used for gathering [thermal dataset](https://public.roboflow.com/object-detection/thermal-dogs-and-people)

## Articles & repos
* [Fire and smoke detection with Keras and Deep Learning by pyimagesearch](https://www.pyimagesearch.com/2019/11/18/fire-and-smoke-detection-with-keras-and-deep-learning/) - dataset collected by scraping Google images (provides link to dataset with  1315 fire images), binary Fire/Non-fire classification with tf2 & keras sequential CNN, achieve 92% accuracy, concludes that better datasets are required
* [Fire Detection from scratch using YOLOv3](https://medium.com/@b117020/fire-detection-using-neural-networks-4d52c5cd55c5) - discusses annotation using LabelImg, using Google drive and Colab, deployment via Heroku and viz using Streamlit
* [fire-detect-yolov4](https://github.com/gengyanlei/fire-detect-yolov4) - training of Yolo v4 model
* [midasklr/FireSmokeDetectionByEfficientNet](https://github.com/midasklr/FireSmokeDetectionByEfficientNet) - Fire and smoke classification and detection using efficientnet, pytorch, visualizes the activation map
* [arpit-jadon/FireNet-LightWeight-Network-for-Fire-Detection](https://github.com/arpit-jadon/FireNet-LightWeight-Network-for-Fire-Detection) - A Specialized Lightweight Fire & Smoke Detection Model for Real-Time IoT Applications (e.g. on RPi), accuracy approx. 95%. Paper https://arxiv.org/abs/1905.11922v2
* [tobybreckon/fire-detection-cnn](https://github.com/tobybreckon/fire-detection-cnn) - links to a couple of datasets
* [EmergencyNet](https://github.com/ckyrkou/EmergencyNet) - identify fire and other emergencies from a drone
* [Fire Detection using CCTV images — Monk Library Application](https://medium.com/towards-artificial-intelligence/fire-detection-using-cctv-images-monk-library-application-242df1fca2b9) - keras classifier on kaggle datasets, mobilenet-v2, densenet121 and densenet201
* [fire-detection-cnn](https://github.com/tobybreckon/fire-detection-cnn) - automatic detection of fire pixel regions in video (or still) imagery within real-time bounds. maximal accuracy of 0.93 for whole image binary fire detection (1), with 0.89 accuracy within our superpixel localization framework can be achieved
* [Early Fire detection system using deep learning and OpenCV](https://towardsdatascience.com/early-fire-detection-system-using-deep-learning-and-opencv-6cb60260d54a) - customized InceptionV3 and CNN architectures for indoor and outdoor fire detection. 980 images for training and 239 images for validation, training accuracy of 98.04 and a validation accuracy of 96.43, openCV used for live detection on webcam - code and datasets (already referenced here) on https://github.com/jackfrost1411/fire-detection

## Datasets
* [FireNET](https://github.com/OlafenwaMoses/FireNET) - approx. 500 fire images with bounding boxes
* [Fire Detection from CCTV on Kaggle](https://www.kaggle.com/ritupande/fire-detection-from-cctv) - images and video, images are extracted from video, relatively small dataset. Quite relevant to current task as have videos to test on.
* [cair/Fire-Detection-Image-Dataset](https://github.com/cair/Fire-Detection-Image-Dataset) - This dataset contains normal images and images with fire, and is highly unbalanced to reciprocate real world situations.
* [mivia Fire Detection Dataset](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/) - approx. 30 videos
* [USTC smoke detection](http://smoke.ustc.edu.cn/datasets.htm) - links to various sources that provide videos of smoke
* fire/not-fire dataset in the pyimagesearch article can be downloaded. Note that there are many images of fire scenes that do not contain actual fire, but burnt out homes for example.
* [FIRE Dataset on Kaggle](https://www.kaggle.com/phylake1337/fire-dataset) - 755 outdoor fire images and 244 non-fire images. Many glossy images, representative?
* [Fire Image Data Set for Dunnings 2018 study](https://collections.durham.ac.uk/files/r2d217qp536#.X2rv1ZNKidb) - PNG still image set
* [Fire Superpixel Image Data Set for Samarth 2019 study](https://collections.durham.ac.uk/files/r10r967374q#.X2rv1pNKidb) - PNG still image set

## Fire safety references
* Locate reference covering the different kinds of fires in the home, common scenarios & interventions
* Safety/accuracy standards for fire detectors, including ROC characteristics

## Fires in the home
* Common causes including cigarettes left smouldering, candles, electrical failures, chip pan fires
* A large number of factors affect the nature of the fire, primarily the fuel and oxygenation, but also where the fire is, middle of the room/against a wall, thermal capacity of a room, the walls, ambient temperature, humidity, contaminants on the material (dust, oil based products, emollients etc)
* To put out a fire a number of retardants are considered - water (not on electrical or chip pan), foam, CO2, dry powder
* In electrical fires the electricity supply should first be isolated
* Reducing ventillation, e.g. by closing doors, will limit fire
* Smoke itself is a strong indicator of the nature of the fire
* Read https://en.m.wikipedia.org/wiki/Fire_triangle and https://en.m.wikipedia.org/wiki/Combustion

## Comments
* Firenet is a VERY common name for model, do not use
