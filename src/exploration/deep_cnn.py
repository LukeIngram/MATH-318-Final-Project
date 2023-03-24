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
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape = None, 
        alpha = 1.0, # using default input width
        include_top = True, #include fully-connected input layer
        weights = 'imagenet', # Default weights
        input_tensor = None, # using default input tensor structure
        pooling = None, # not using this feaure
        classes = 1000, # default
        classifier_activation = 'softmax' #specify activation function of output layer
    )   

    for layer in model.layers: 
        layer.trainable = False # Do not overwrite existing weights

    # Create output layer TODO 

    final_layer = model.layers[-2].output

    output_layer = tf.keras.layers.Dense(144,activation='relu')(final_layer)
    output_layer = tf.keras.layers.Dense(72,activation='relu')(final_layer)
    output_layer = tf.keras.layers.Dense(6,activation='softmax')(final_layer)

    # add it back to model
    model = tf.keras.Model(inputs = model.layers[0].input,outputs = output_layer)

    # Compile final model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='SGD', # Stochastic- Gradient - Descent
        metrics = ['accuracy']
    )

    # Convert to float for tensor conversion 
    train_X = np.asarray(list(train_X),dtype='float32')
    test_X = np.asarray(list(test_X),dtype='float32')
    val_X = np.asarray(list(val_X),dtype='float32')


    training_history = model.fit(train_X,train_Y,epochs = 30,validation_data=(val_X,val_Y))

    model.evaluate(test_X,test_Y) 

    # SAVE FOR LATER 
    model.save('models/mobileNetV2.h5')

    # Generate plot

    





if __name__ == '__main__': 
    main()