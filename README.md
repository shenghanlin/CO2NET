# CO2NET
Deep learning for characterizing CO2 migration in time-lapse seismic images[pdf](https://www.sciencedirect.com/science/article/abs/pii/S0016236122036304)


## Introduction

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

 To be updated......
