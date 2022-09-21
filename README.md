# face-shape-model

## 1. Face Shape

### 1.1. Dataset
[Face Shape Dataset - NITEN LAMA](https://www.kaggle.com/datasets/niten19/face-shape-dataset)

<br>  

### 1.2. Model : EfficientNet - B5
reference code: 
[Face Shape - EfficientNetB5 - Acc 68%](https://www.kaggle.com/code/tqtrinh/face-shape-efficientnetb5-acc-68)

<br>  

### 1.3. Accuracy : 80%
![image](https://user-images.githubusercontent.com/61939286/190403874-b3ec842c-0d19-403d-b852-681277cc705c.png)

<br>  

### 1.4. Predict Test
| |한소희|박은빈|카리나|김다미|김태리|
|--|----|---|---|---|---|
|IMG|![image](https://user-images.githubusercontent.com/61939286/190890559-016e7202-36cc-4d48-8817-15264b591f0d.png)|![박은빈](https://user-images.githubusercontent.com/61939286/190405695-b80983f7-65dc-4b36-b625-e81d53c02062.png)|![카리나](https://user-images.githubusercontent.com/61939286/190889494-04d9ddf5-fcb1-44a1-950d-92866e41e5ef.png)|![김다미](https://user-images.githubusercontent.com/61939286/190409030-c5b174a3-9e64-4863-b89a-04a0ec7bc9e3.png)|![image](https://user-images.githubusercontent.com/61939286/190889566-f914f4c8-3380-49af-b159-adfaead595d3.png)|
|Real|Heart|Oblong|Oval|Round|Square|
|Predict|Heart|Oblong|Oval|Round|Square|

<br>  
<br>  

## 2. Facial Features

### 2.1 Framework
[Google Mediapipe](https://github.com/google/mediapipe) - Face Mesh

<br>  

### 2.2 Analysis
|mid|chin|philtrum|
|--|----|---|
|<img src="https://user-images.githubusercontent.com/61939286/191513217-7e8b0438-5aca-4dc6-a85b-d382e1aa2766.png" width="300">|<img src="https://user-images.githubusercontent.com/61939286/191512737-8c74f5b7-3b03-4716-af39-1588c49c7658.png" width="300">|<img src="https://user-images.githubusercontent.com/61939286/191512801-677f2a4b-4bb7-4663-b14d-2d7a9fc3badd.png" width="300">|

<br>  

### 2.3 Predict Test
<img src="https://user-images.githubusercontent.com/61939286/191514789-78ffac3e-0448-4208-ac76-8923f05aa5ca.png" width="60%">

```
face ratio :  4.411731675378876
is_wide_margin =  False
is_long_mid =  False
is_long_chin :  False
is_long_philtrum =  True
```


<br>  

## 3. Hair Style Consult

### 3.1 Test Report
<img src="https://user-images.githubusercontent.com/61939286/191519174-e415430b-313a-4ec5-a704-960722b3ec30.png" width="70%">


