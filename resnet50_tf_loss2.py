from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
import numpy as np 
import tensorflow as tf 

from sklearn.metrics import log_loss

from load_cifar10 import load_cifar10_data
from scipy.io import savemat

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    The identity_block is the block that has no conv layer at shortcut
    Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x

def resnet50_model(img_rows, img_cols, color_type=1, num_classes=None):
    """
    Resnet 50 Model for Keras
    Model Schema is based on 
    https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
    ImageNet Pretrained Weights 
    https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of class labels for our classification task
    """

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type))
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols))

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # Fully Connected Softmax Layer
    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    # Create model
    model = Model(img_input, x_fc)

    # Load ImageNet pre-trained data 
    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = 'imagenet_models/resnet50_weights_th_dim_ordering_th_kernels.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = 'weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

    model.load_weights(weights_path)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)

    # Create base model
    model = Model(img_input, x_newfc)

    # x_newfc = Flatten()(x_newfc)
    # x_newfc = Dense(num_classes, activation='softmax', name='fc10')(x_newfc)

    # Create another model with our customized softmax
    # model = Model(img_input, x_newfc)

    # # Learning rate is changed to 0.001
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
  
    return model

def add_new_last_layer(base_model, num_classes):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
#   FC_SIZE = 1024
#   x = Dense(FC_SIZE, activation='relu')(base_model.output) # new FC layer, random init
  x = Flatten()(base_model.output)
  predictions = Dense(num_classes, activation='softmax')(x) # new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model

# def setup_to_transfer_learn(model):
#   """Freeze all layers and compile the model"""

#   for cnt in range(len(model.layers)-1):
#       model.layers[cnt].trainable = False
# #   for layer in base_model.layers:
# #     layer.trainable = False
# # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#   model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
# #   model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
#   model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])


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
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10
    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 10 
    batch_size = 16 
    nb_epoch = 5000

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)
    print("Data loaded.")

    # Define training parameters 
    num_samples = X_train.shape[0]
    num_test_samples = X_valid.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    num_batches_test = int(np.ceil(num_test_samples / batch_size))

    # Load our model
    K.set_learning_phase(1)
    base_model = resnet50_model(img_rows, img_cols, channel, num_classes)
    model = add_new_last_layer(base_model, num_classes)
    print("Model loaded.")

    # Conduct transfer learning 
    setup_to_transfer_learn(model, base_model)
    print("Setup for transfer done.")


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

    # Define accuracy 
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_sum(correct_prediction)

    # Create a summary to monitor the accuracy 
    tf.summary.scalar("accuracy", accuracy)

    # Merge all summaries and write them out 
    merged = tf.summary.merge_all()
  
    with tf.Session() as sess: 
        train_writer = tf.summary.FileWriter('train', sess.graph)
        test_writer = tf.summary.FileWriter('test')

        sess.run(tf.global_variables_initializer())
        print("Global variables initialized.")

        train_accuracy_history, test_accuracy_history = [], []
        
        for epoch in range(nb_epoch): 
            print('epoch %d: ' % (epoch))

            for i in range(num_batches):
                #for i in range(5):  
                # Grab batch of training data 
                batch_indices = np.random.randint(num_samples, size=batch_size)
                batch_x = X_train[batch_indices]
                batch_y = Y_train[batch_indices]
                sess.run(update, feed_dict={x: batch_x, y_: batch_y})
                print('step %d: ' % (i))

            # TO-DO: find way to save accuracy efficiently 
            # summary, train_accuracy = sess.run([merged, accuracy], feed_dict={x: X_train, y_: Y_train})
            # print('step %d, training accuracy %g' % (epoch, train_accuracy))
            # train_writer.add_summary(summary, epoch)

            # summary, test_accuracy = sess.run([merged, accuracy], feed_dict={x: X_test, y_: Y_test})
            # print('step %d, training accuracy %g' % (epoch, test_accuracy))
            # test_writer.add_summary(summary, epoch)
            

            # Compute the training accuracy 
            total_train_accuracy = 0.0 
            for batch_idx in range(num_batches):
                if batch_idx == num_batches - 1:
                    batch_x = X_train[batch_idx*batch_size:num_samples, :, :, :]
                    batch_y = Y_train[batch_idx*batch_size:num_samples]
                else:
                    batch_x = X_train[batch_idx*batch_size : (batch_idx+1)* batch_size, :, :, :]
                    batch_y = Y_train[batch_idx*batch_size : (batch_idx+1)* batch_size]

                train_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y})
                total_train_accuracy = total_train_accuracy + train_accuracy
            
            final_train_accuracy  = total_train_accuracy / (float(num_samples))
            print("Train Accuracy: %f" % (final_train_accuracy))
            train_accuracy_history.append(final_train_accuracy)

            # Compute the testing accuracy 
            total_test_accuracy = 0.0 

            for batch_idx in range(num_batches_test):
                if batch_idx == num_batches_test - 1:
                    batch_x = X_valid[batch_idx*batch_size :num_test_samples, :, :, :]
                    batch_y = Y_valid[batch_idx*batch_size :num_test_samples]
                else:
                    batch_x = X_valid[batch_idx*batch_size : (batch_idx+1)* batch_size, :, :, :]
                    batch_y = Y_valid[batch_idx*batch_size : (batch_idx+1)* batch_size]

                test_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y})
                total_test_accuracy = total_test_accuracy + test_accuracy
            
            final_test_accuracy  = total_test_accuracy / (float(num_test_samples))
            print("Test Accuracy: %f" % (final_test_accuracy))
            test_accuracy_history.append(final_test_accuracy)

            save_to_mat = {}
            save_to_mat['training_accuracies'] = train_accuracy_history
            save_to_mat['test_accuracies'] = test_accuracy_history
            savemat("./summary_results.mat", {'training_summary':save_to_mat})

    # Start transfer learning. 
    # model.fit(X_train, Y_train,
    #           batch_size=batch_size,
    #           nb_epoch=nb_epoch,
    #           shuffle=True,
    #           verbose=1,
    #           validation_data=(X_valid, Y_valid),
    #           )

    # # Make predictions
    # predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
