# Traffic-sign-recognition
### Description:

This project is a part of the [Harvard CS50AI](https://cs50.harvard.edu/ai/2023/) course.
The project is a program which uses TensorFlow framework to build a neural network to classify images. 
Main parts of a project are:

Learning (learn.py)
- preparation data for modeling (data are images of traffic signes divided into folders)
- creation and training a model

Recognition (predict.py)
- preparation data for recognition (load images from folder)
- recognition and printing a result

As a data set for the traning [German Traffic Sign Recognition Benchmark](https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip) had been taken<br>
As an images for recognition screenshots from Google Maps (Germany, Berlin) had been taken

***
### Model parameters

Main goal of choosing parameters is to find best accuracy (within couple minutes of training).<br>
As the default following setup is chosen:

| Parameter | Value |
|---|---|
| Number of convolution/pooling layers | 1 |
| Number of convolution filters | 32 |
| Convolution kernel size | 3x3 |
| Pool size | 2x2 |
| Number of hidden layers | 1 |
| Size of hidden layer | 128 |
| Dropout | 0.5 |
| **Accuracy** | **0.96** |

***
### Experiment results

| Number of convolution/pooling layers | Accuracy |
|---|---|
| 1 | 0.96 |
| 2 | 0.98 |
| 3 | 0.97 |

-> 2% accuracy increase between 1 and 2 convolutions is reached because neural network becomes less sensitive to variation (picture taken from slightly different angles).

***

| Number of convolution filters | Accuracy |
|---|---|
| 2 | 0.91 |
| 4 | 0.95 |
| 8 | 0.96 |
| 16 | 0.96 |
| 32 | 0.96 |
| 64 | 0.96 |
| 128 | 0.97 |

-> Exponential increase of filters amount increases accuracy, which has asymptotic behavior.

***

| Convolution kernel size | Accuracy |
|---|---|
| 3x3 | 0.96 |
| 9x9 | 0.96 |

-> Kernel size does not strongly affect on accuracy.

***

| Pool size | Accuracy |
|---|---|
| 1x1 | 0.97 |
| 2x2 | 0.96 |
| 4x4 | 0.92 |

-> Bigger pooling kernel decrease accuracy (picture became more pixelized). But also computation time decrease.

***

| Number of hidden layers | Accuracy |
|---|---|
| 1 | 0.96 |
| 2 | 0.97 |
| 3 | 0.96 |

-> In this case with 128 neuron density number of layers canging does no effect.

***

| Size of hidden layer | Accuracy |
|---|---|
| 64 | 0.89 |
| 128 | 0.96 |
| 256 | 0.96 |
| 512 | 0.97 |

-> Exponential increase of neurons amount increases accuracy, which has asymptotic behavior.

***

| Dropout | Accuracy |
|---|---|
| 0 | 0.94 |
| 0.1 | 0.96 |
| 0.25 | 0.96 |
| 0.4 | 0.95 |
| 0.5 | 0.96 |
| 0.6 | 0.96 |
| 0.75 | 0.91 |
| 0.9 | 0.70 |
| 1 | ValueError :) |

-> Most optimal amount of dropout is half. Decreasing of dropout has risk of overfitting. Dropping out too many neurons make accuracy worse.

***
### Additional experiment while compiling

| Loss function | Accuracy |
|---|---|
| categorical_crossentropy | 0.96 |
| binary_crossentropy | 0.94 |

-> Binary crossentropy should work better with binary classification, not multi-class classification.

***

| Batch size | Batches number | Accuracy |
|---|---|---|
| 1 | 15984 | 0.94 |
| 32 (default) | 500 | 0.97 |
| 1000 | 16 | 0.77 |
| 10000 | 2 | 0.21 |

-> Lack of memory or complexity of model may be reason of decreasing of batch size. But deafult size is best for this case.

***

| Epochs | Accuracy |
|---|---|
| 10 | 0.96 |
| 20 | 0.97 |

-> Increasing amount of epochs after 10 doesn't make much sence with this set of settings.

***
### Conclusion
Initial set of parameters is considered to be optimal, except:
- number of convolution/pooling layers
- number of hidden layers

Both numbers are set to 2 for best performance (0.98 of accurcy):

```bash
Epoch 1/10
500/500 [==============================] - 8s 14ms/step - loss: 2.4862 - accuracy: 0.2994
Epoch 2/10
500/500 [==============================] - 7s 14ms/step - loss: 0.9245 - accuracy: 0.7020
Epoch 3/10
500/500 [==============================] - 7s 14ms/step - loss: 0.4144 - accuracy: 0.8681
Epoch 4/10
500/500 [==============================] - 7s 14ms/step - loss: 0.2456 - accuracy: 0.9256
Epoch 5/10
500/500 [==============================] - 7s 14ms/step - loss: 0.1860 - accuracy: 0.9454
Epoch 6/10
500/500 [==============================] - 7s 14ms/step - loss: 0.1340 - accuracy: 0.9590
Epoch 7/10
500/500 [==============================] - 7s 14ms/step - loss: 0.1063 - accuracy: 0.9668
Epoch 8/10
500/500 [==============================] - 7s 14ms/step - loss: 0.0922 - accuracy: 0.9735
Epoch 9/10
500/500 [==============================] - 7s 14ms/step - loss: 0.0816 - accuracy: 0.9764
Epoch 10/10
500/500 [==============================] - 7s 14ms/step - loss: 0.0612 - accuracy: 0.9822
333/333 - 1s - loss: 0.0590 - accuracy: 0.9861 - 1s/epoch - 3ms/step
```
Prediction for all images (except 4th) had been chosen with 100% confidence.<br>
The 4th image didn't present in training set, which caused less confidence.<br>
The 5th image had noise (sticker on a sign), but it didn't affect confidence:
```bash
1/1 [==============================] - 0s 133ms/step

img1.png is "Speed limit (30km/h)" sign with a confidence of 100.00%
img2.png is "No entry" sign with a confidence of 100.00%
img3.png is "Priority road" sign with a confidence of 100.00%
img4.png is "Turn right ahead" sign with a confidence of 75.91%
img5.png is "Stop" sign with a confidence of 100.00%
```