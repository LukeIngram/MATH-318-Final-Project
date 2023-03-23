# deep_cnn.py
# Exploration of adapting google's mobilenet V2 deep cnn to classify our data

import cv2 
import tensorflow as tf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def main(): 
    #Load data 
    dirName = "data/archive/zero-indexed-files.txt"
    imgPath = "data/archive/Garbage_classification/load/"

    df = pd.read_csv(dirName,sep=' ')
   
    df['image'] = imgPath + df['image'].astype(str)
    df['image'] = df['image'].apply(lambda x: cv2.imread(x))
    print(df.head()) #DEBUG
    
    train_X,test_X,train_Y,test_Y = train_test_split(df['image'],df['class'],
                                                     test_size=0.20,random_state=42,stratify=df['class'])
    
    train_X,val_X,train_Y,val_Y = train_test_split(train_X,test_Y,
                                                   test_size=0.25,random_state=42,stratify=test_Y)
    

    # Normalize data 
    train_X /= 255.0 
    test_X /= 255.0

    # Fetch network from keras, & define custom params. 
    # This is a pre-trained model,bbut we remove the last layer & train it ourselves to fit out problem
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape = (512,384,3),  
        alpha = 1.0, # using default input width
        include_top = True, #include fully-connected input layer
        weights = 'imagenet', # Default weights
        input_tensor = None, # using defualt input tensor structure
        pooling = None, # not using this feature
        classes = 1000, # default
        classifier_activation = 'softmax' #specify activation function of output layer
    )   

    # Create output layer TODO 
    output_layer = tf.keras.layers.Dense(6,activation='softmax')(model.layers[-1].output)

    # add it back to model
    model = tf.keras.Model(inputs = model.layers[0].input,outputs = output_layer)

    # Compile final model
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy,
        optimizer=tf.keras.optimizers.SGD, # Stochastic- Gradient - Descent
        metrics = ["accuracy"]
        )
    

    training_history = model.fit(train_X,train_Y,epochs = 300,validation_data=(val_X,val_Y))

    model.evaluate(test_X,test_Y) 

    





if __name__ == '__main__': 
    main()