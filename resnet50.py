"""Perform transfer learning on ResNet50
pretrained on ImageNet to CIFAR-10.

Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf

Overcoming Catastrophic Forgetting
https://arxiv.org/abs/1612.00796
"""

import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD
from load_cifar10 import load_cifar10_data
import numpy as np
import os


# Parameters 
img_rows, img_cols = 32, 32 # Resolution of inputs
channel = 3
num_classes = 10 
batch_size = 32
nb_epoch = 10
num_layers_to_freeze = 10 
FC_SIZE = 1024 

def add_new_last_layer(base_model, num_classes):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
#   x = base_model.output 
#   x = MaxPooling2D(x) # take the output from the final max pooling layer 
#   x = base_model.output
#   x = GlobalAveragePooling2D()(x)
#   x = GlobalMaxPooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(base_model.output) # new FC layer, random init
  predictions = Dense(num_classes, activation='softmax')(x) # new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model


def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False

#   sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
#   model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


if __name__=="__main__": 
    # Load CIFAR-10 data 
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)
    print("data loaded")
    print X_train.shape, Y_train.shape
    
    # Load ResNet50 model without the top FC layer 
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=None, pooling="max")

    # Add last layer 
    model = add_new_last_layer(base_model, num_classes)
    print("model loaded")

    # Conduct transfer learning 
    setup_to_transfer_learn(model, base_model)

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10_resnet_model.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate decaying.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                verbose=1,
                                save_best_only=True)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-6)
    tb = keras.callbacks.TensorBoard(log_dir='./logs',
                                    histogram_freq=0,
                                    write_graph=True,
                                    write_images=True)
    callbacks = [checkpoint, lr_reducer, tb]

    # Start training (without fine-tuning)
    # TO-DO: implement with fit_generator for real-time data augmentation 
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              callbacks=callbacks,
              validation_data=(X_valid, Y_valid),
            )

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    # score = log_loss(Y_valid, predictions_valid)






