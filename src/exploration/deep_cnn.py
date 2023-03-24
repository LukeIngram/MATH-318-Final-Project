# deep_cnn.py
# Exploration of adapting google's mobilenet V2 deep cnn to classify our data

import cv2 
import tensorflow as tf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os 




def main(): 
    #Load data 
    dirName = "data" + os.sep + "archive" + os.sep + "zero-indexed-files.txt"
    imgPath = "data" + os.sep + "archive" + os.sep + "Garbage_classification" + os.sep +  "load" + os.sep

    df = pd.read_csv(dirName,sep=' ')
   
    df['image'] = imgPath + df['image'].astype(str)
    df['image'] = df['image'].apply(lambda x: cv2.resize(cv2.imread(x),(224,224)))
    print(df.head()) #DEBUG
    
    train_X,test_X,train_Y,test_Y = train_test_split(df['image'],df['class'],
                                                     test_size=0.33,random_state=42,stratify=df['class'])

    
    train_X,val_X,train_Y,val_Y = train_test_split(train_X,train_Y,
                                                   test_size=0.20,random_state=42,stratify=train_Y)
    

    ## Normalize data 
    train_X = train_X/255.0 
    test_X = test_X/255.0
    val_X = val_X/255.0

    # Fetch network from keras, & define custom params. 
    # This is a pre-trained model,bbut we remove the last layer & train it ourselves to fit out problem
    mobile_layer = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape = (224,224,3), # our images are in the shape
        alpha = 1.0, # using default input width
        include_top = False, #Don't include fully-connected input layer, we specify our own later
        weights = 'imagenet', # Default weights
        input_tensor = None, # using default input tensor structure
        pooling = None, # not using this feature
        classes = 1000, # default
        classifier_activation = 'softmax' #specify activation function of output layer
    )   

    mobile_layer.trainable = False

    model = tf.keras.models.Sequential()

    model.add(tf.keras.Input(shape=(224,224,3))) # add our input layer
    model.add(mobile_layer)# attach the pretrained portion of the model

    # add our custom layers
    model.add(tf.keras.layers.GlobalAvgPool2D()) # add an additional pooling layer
    model.add(tf.keras.layers.Dense(6,activation='softmax')) # add softmax output layer

    # Compile final model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam', # adam optimizer for Stochastic Gradient Descent
        metrics = ['accuracy']
    )

    # Convert to float for tensor conversion 
    train_X = np.asarray(list(train_X),dtype='float32')
    test_X = np.asarray(list(test_X),dtype='float32')
    val_X = np.asarray(list(val_X),dtype='float32')


    training_history = model.fit(train_X,train_Y,epochs = 30,validation_data=(val_X,val_Y))

    model.evaluate(test_X,test_Y) 

    # SAVE FOR LATER 
    model.save('models/mobileNetV2_2.h5')

    # Generate plot

    





if __name__ == '__main__': 
    main()