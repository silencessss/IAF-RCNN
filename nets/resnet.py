#-------------------------------------------------------------#
#   ResNet50
#-------------------------------------------------------------#
from __future__ import print_function

from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Activation, Add, AveragePooling2D, Conv2D, BatchNormalization,
                                     MaxPooling2D, TimeDistributed,
                                     ZeroPadding2D)


def identity_block(name, input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters

    conv_name_base = name+'res' + str(stage) + block + '_branch'
    bn_name_base = name+'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), kernel_initializer=RandomNormal(stddev=0.02), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(trainable=False, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer=RandomNormal(stddev=0.02), name=conv_name_base + '2b')(x)
    x = BatchNormalization(trainable=False, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), kernel_initializer=RandomNormal(stddev=0.02), name=conv_name_base + '2c')(x)
    x = BatchNormalization(trainable=False, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(nameset,input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    conv_name_base = nameset+'res' + str(stage) + block + '_branch'
    bn_name_base = nameset+'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, kernel_initializer=RandomNormal(stddev=0.02),
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(trainable=False, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer=RandomNormal(stddev=0.02),
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(trainable=False, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), kernel_initializer=RandomNormal(stddev=0.02), name=conv_name_base + '2c')(x)
    x = BatchNormalization(trainable=False, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_initializer=RandomNormal(stddev=0.02),
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(trainable=False, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(inputs,name):
    #-----------------------------------#
    #   input shape: (600,600,3)
    #-----------------------------------#
    img_input = inputs

    # 600,600,3 -> 300,300,64
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name=name+'conv1')(x)
    x = BatchNormalization(trainable=False, name=name+'bn_conv1')(x)
    x = Activation('relu')(x)

    # 300,300,64 -> 150,150,64
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # 150,150,64 -> 150,150,256
    x = conv_block(name,x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(name, x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(name, x, 3, [64, 64, 256], stage=2, block='c')

    # 150,150,256 -> 75,75,512
    x = conv_block(name,x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(name, x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(name, x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(name, x, 3, [128, 128, 512], stage=3, block='d')

    # 75,75,512 -> 38,38,1024
    x = conv_block(name,x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(name, x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(name, x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(name, x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(name, x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(name, x, 3, [256, 256, 1024], stage=4, block='f')

    # output feature map: (38,38,1024)
    return x

def identity_block_td(nameset, input_tensor, kernel_size, filters, stage, block):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = nameset+'res' + str(stage) + block + '_branch'
    bn_name_base = nameset+'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(trainable=False), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(trainable=False), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(BatchNormalization(trainable=False), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x

def conv_block_td(nameset, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = nameset+'res' + str(stage) + block + '_branch'
    bn_name_base = nameset+'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), strides=strides, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(trainable=False), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(trainable=False), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(BatchNormalization(trainable=False), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(Conv2D(nb_filter3, (1, 1), strides=strides, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(BatchNormalization(trainable=False), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def classifier_layers(x,nameset):
    # num_rois, 14, 14, 1024 -> num_rois, 7, 7, 2048
    x = conv_block_td(nameset, x, 3, [512, 512, 2048], stage=5, block='a', strides=(2, 2))
    # num_rois, 7, 7, 2048 -> num_rois, 7, 7, 2048
    x = identity_block_td(nameset, x, 3, [512, 512, 2048], stage=5, block='b')
    # num_rois, 7, 7, 2048 -> num_rois, 7, 7, 2048
    x = identity_block_td(nameset, x, 3, [512, 512, 2048], stage=5, block='c')
    # num_rois, 7, 7, 2048 -> num_rois, 1, 1, 2048
    x = TimeDistributed(AveragePooling2D((7, 7)), name=nameset+'avg_pool')(x)

    return x
