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
    df['image'] = df['image'].apply(lambda x: cv2.resize(cv2.imread(x),(224,224)))
    print(df.head()) #DEBUG
    
    train_X,test_X,train_Y,test_Y = train_test_split(df['image'],df['class'],
                                                     test_size=0.20,random_state=42,stratify=df['class'])

    
    train_X,val_X,train_Y,val_Y = train_test_split(train_X,train_Y,
                                                   test_size=0.20,random_state=42,stratify=train_Y)
    

    ## Normalize data 
    train_X = train_X/255.0 
    test_X = test_X/255.0

    # Fetch network from keras, & define custom params. 
    # This is a pre-trained model,bbut we remove the last layer & train it ourselves to fit out problem
    model = tf.keras.applications.VGG19(
        input_shape = None, 
        #alpha = 1.0, # using default input width
        include_top = True, #include fully-connected input layer
        weights = 'imagenet', # Default weights
        input_tensor = None, # using defualt input tensor structure
        pooling = None, # not using this feature
        classes = 1000, # default
        classifier_activation = 'softmax' #specify activation function of output layer
    )   

    for layer in model.layers: 
        layer.trainable = False # Do not overwrite existing weights

    # Create output layer TODO 

    final_layer = model.layers[-2].output

    output_layer = tf.keras.layers.Dense(72,activation='relu')(final_layer)
    output_layer = tf.keras.layers.Dense(36,activation='relu')(final_layer)
    output_layer = tf.keras.layers.Dense(6,activation='softmax')(final_layer)

    # add it back to model
    model = tf.keras.Model(inputs = model.layers[0].input,outputs = output_layer)

    # Compile final model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam', # Stochastic- Gradient - Descent
        metrics = ["accuracy"]
        )

    
    train_X = np.asarray(list(train_X),dtype='float32')
    #train_Y = np.asarray(list(train_Y))
    test_X = np.asarray(list(test_X),dtype='float32')
    #test_Y = np.asarray(list(test_Y))
    val_X = np.asarray(list(val_X),dtype='float32')
    #val_Y = np.asarray(val_Y)


    training_history = model.fit(train_X,train_Y,epochs = 30,validation_data=(val_X,val_Y))

    model.evaluate(test_X,test_Y) 

    # SAVE FOR LATER 
    model.save('models/VGG16.h5')






if __name__ == '__main__': 
    main()