# CO2NET :shipit:
Deep learning for characterizing CO2 migration in time-lapse seismic images [[PDF Version]](https://www.sciencedirect.com/science/article/abs/pii/S0016236122036304)



<!-- more -->

### Preparation

```c++
python.__version__  '3.6.2'
tf.__version__      '1.13.1'
keras.__version__   '2.3.1'
```


## Data

I compress the data and then reduce the number of validation datasets because validation datasets do not affect the training. I have uploaded all these datasets to GDrive.

https://drive.google.com/drive/folders/1hHDkq3qyqNUU3V221OaWHGTcn458YyNg?usp=sharing

### Training


```python
python train.py
```

### Prediction


```python
python apply_field.py  name #### Name is the name of the model used for prediction 
                            #### field, Valid, FeatureMap are three programs for predicting the field data, the synthetic data, and the Feature Map.
```

## Citation
```
@article{SHENG2022CO2,
title = {Deep learning for characterizing CO2 migration in time-lapse seismic images},
journal = {Fuel},
pages = {126806},
year = {2022},
issn = {0016-2361},
doi = {https://doi.org/10.1016/j.fuel.2022.126806},
url = {https://www.sciencedirect.com/science/article/pii/S0016236122036304},
author = {Hanlin Sheng and Xinming Wu and Xiaoming Sun and Long Wu},
keywords = {Seismic data, Seismic interpretation, Deep learning, CO characterization, Convolutional neural network, CO migration},
abstract = {Time-lapse (or 4-D) seismic data play an important role in monitoring the spatial CO2 distribution during and after the injection period. However, traditional interpretation or prediction of CO2 distribution is time-consuming and might be sensitive to the quality of 4D seismic data. To solve these problems, we propose a deep-learning-based method to efficiently and accurately characterize CO2 plumes in time-lapse seismic data. We first introduce a workflow to build 3-D realistic impedance models containing CO2 plumes with various shapes, sizes, and locations. From the impedance models, we then simulate synthetic seismic datasets and automatically obtain the corresponding CO2 label volumes. We extract real noise from field seismic datasets and add the noise to the synthetic ones to make them more realistic. We further construct a diverse and realistic training dataset with the combination of synthetic data containing CO2 plumes and real data without CO2 plumes that are randomly cropped from field seismic data before CO2 injection. We finally utilize the training datasets without any human labeling to train a 3-D deep U-shape convolutional neural network for detecting CO2 plumes in the Sleipner time-lapse seismic images. Compared to traditional interpretation methods that take several days or even weeks, our method takes only 29 s using one graphics processing unit (GPU) to predict CO2 plumes in a 512*512*256 seismic volume. Besides, our CO2 prediction can achieve 95.8% accuracy (compared to the manual interpretation) and could distinguish reflections of CO2 plumes from the ones of pre-existing fluids, thin layers, and noise. To more accurately characterize the CO2 plumes migration, we use dynamic image warping to compute relative shifts that register the time-lapse seismic volumes before and after CO2 injection and then apply the same shifts to the predicted CO2 plumes. By doing this, we are able to reduce the inconsistencies that may be introduced by acquisition, processing, push-down effect (velocity decrease by injected CO2), and pull-up effect (wavelet distortion), which is helpful to more accurately characterize the CO2 plumes migration.}
}
```
