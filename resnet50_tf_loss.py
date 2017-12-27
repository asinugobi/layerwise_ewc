"""Perform transfer learning on ResNet50
pretrained on ImageNet to CIFAR-10.

Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf

Overcoming Catastrophic Forgetting
https://arxiv.org/abs/1612.00796
"""

import keras 
import tensorflow as tf
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, Flatten,  GlobalMaxPooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD
from load_cifar10 import load_cifar10_data
from keras.datasets import cifar10
import numpy as np
import os


# Parameters 
img_rows, img_cols = 224, 224 # Resolution of inputs
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

  x = base_model.layers[-2].output
  predictions = Dense(num_classes, activation='softmax')(x) # new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model


def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for cnt in range(len(model.layers)-1):
      model.layers[cnt].trainable = False
#   for layer in base_model.layers:
#     layer.trainable = False
# sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
#   model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


def setup_to_finetune(model):
  """Freeze the bottom num_layers_to_freeze and retrain the remaining top layers.

  note: num_layes_to_freeze corresponds to the top 2 ResNet blocks

  Args:
    model: keras model
  """
  for layer in model.layers[:num_layers_to_freeze]:
     layer.trainable = False
  for layer in model.layers[num_layers_to_freeze:]:
     layer.trainable = True
   
   # We want to avoid compiling model in Keras. Instead, we will use TF. 

#   model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

if __name__=="__main__": 
    # Load CIFAR-10 data 
    x_train, y_train, x_test, y_test = load_cifar10_data(img_rows, img_cols)
    print("Training data shape: ",  x_train.shape)

    # Define training parameters 
    num_samples = x_train.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))

    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Input image dimensions.
    # We assume data format "channels_last".
    # img_rows = x_train.shape[1]
    # img_cols = x_train.shape[2]
    # channels = x_train.shape[3]

    # if K.image_data_format() == 'channels_first':
    #     img_rows = x_train.shape[2]
    #     img_cols = x_train.shape[3]
    #     channels = x_train.shape[1]
    #     x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
    #     x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    #     input_shape = (channels, img_rows, img_cols)
    # else:
    #     img_rows = x_train.shape[1]
    #     img_cols = x_train.shape[2]
    #     channels = x_train.shape[3]
    #     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    #     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    #     input_shape = (img_rows, img_cols, channels)

    # # Normalize data.
    # x_train = x_train.astype('float32') / 255
    # x_test = x_test.astype('float32') / 255
    # print('x_train shape:', x_train.shape)
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')
    # print('y_train shape:', y_train.shape)

    # # Convert class vectors to binary class matrices.
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    print("data loaded")
    # print x_train.shape, y_train.shape
    
    # Load ResNet50 model without the top FC layer 
    # base_model = ResNet50(include_top=False, weights='imagenet', input_shape=None, pooling="max")
    base_model = ResNet50(include_top=True, weights='imagenet')

    # Add last layer 
    model = add_new_last_layer(base_model, num_classes)
    print("Model loaded.")

    # Conduct transfer learning 
    setup_to_transfer_learn(model, base_model)
    print("Setup for transfer done.")

    # Create TF loss function 
    x = model.input 
    y = model.layers[-1].output
    print(x.shape)
    print(y.shape)

    # _x = np.random.rand()
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                   logits=y)
    loss = tf.reduce_mean(loss)

    print("Loss defined.")

    # Define optimizer 
    update = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    print("Optimizer defined.")

    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())
        print("Global variables initialized.")

        for i in range(num_batches): 
            # Grab batch of training data 
            batch_indices = np.random.randint(num_samples, size=batch_size)
            batch = x_train[batch_indices]

            # Compute model output 
            sess.run(y, feed_dict={x: batch})
            print(i)
            # update.run(feed_dict={x: batch, })




    # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
    #                             cooldown=0,
    #                             patience=5,
    #                             min_lr=0.5e-6)
    # tb = keras.callbacks.TensorBoard(log_dir='./logs',
    #                                 histogram_freq=0,
    #                                 write_graph=True,
    #                                 write_images=True)
    
    # callbacks = [checkpoint, lr_reducer, tb]
    # callbacks = [lr_reducer, tb]

    # Start training (without fine-tuning)
    # TO-DO: implement with fit_generator for real-time data augmentation 
    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           nb_epoch=nb_epoch,
    #           shuffle=True,
    #           verbose=1,
    #           callbacks=callbacks,
    #           validation_data=(x_test, y_test),
    #         )

    # Make predictions
    # predictions_valid = model.predict(x_test, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    # score = log_loss(Y_valid, predictions_valid)






