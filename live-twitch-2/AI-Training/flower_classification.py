# LIBRAIRIES IMPORTATION
import numpy as np
import cv2 as cv
import os
import datetime
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


# DEFINE BUILDING FUNCTION 
def buildModel():
        
    # ResNet50 model
    resnet_50 = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
    for layer in resnet_50.layers:
        layer.trainable = False

    # build the entire model
    x = resnet_50.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x) 
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x) 
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x) 
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x) 
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(5, activation='softmax')(x)
    
    model = Model(inputs = resnet_50.input, outputs = predictions)
        
    return model


# DEFINE TRAINING FUNCTION
def trainModel(model, epochs, optimizer):
    
    batch_size = 32
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # add the TensorBoard callback
    log_dir = "runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    return model.fit(train_generator, validation_data=valid_generator, epochs=epochs, batch_size=batch_size, callbacks = [tensorboard_callback])


# DEFINE EVALUATION FUNCTION
def evaluateModel(model):
    
    test_loss, test_acc = model.evaluate(test_generator)
    
    return test_loss, test_acc


# DEFINE EXPORT FUNCTION 
def exportModel(model):

    model.save('/workspace/saved_model/my_model')
    model = tf.keras.models.load_model('/workspace/saved_model/my_model')
    
    return model


# MAIN 
if __name__ == '__main__':
    
    # data generator
    datagen = ImageDataGenerator()
    
    # define classes name
    class_names = ['daisy','dandelion','roses','sunflowers','tulips']
    
    # training data
    train_generator = datagen.flow_from_directory( 
        directory="/workspace/data-split/train/", 
        classes = class_names,
        target_size=(224, 224),  
        batch_size=32, 
        class_mode="binary", 
    )
    
    # validation data
    valid_generator = datagen.flow_from_directory( 
        directory="/workspace/data-split/val/", 
        classes = class_names,
        target_size=(224, 224), 
        batch_size=32, 
        class_mode="binary", 
    )
    
    # test data
    test_generator = datagen.flow_from_directory( 
        directory="/workspace/data-split/test/", 
        classes = class_names,
        target_size=(224, 224), 
        batch_size=32, 
        class_mode="binary", 
    )
    
    # build the model
    model = buildModel()
    
    # launch the training
    model_history = trainModel(model = model, epochs = 10, optimizer = "Adam")
    
    # evaluate the model
    model_evaluation = evaluateModel(model)
    
    # save and exoport the model in the Object Storage 
    model_export = exportModel(model)
    
    