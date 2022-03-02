import tensorflow
from tensorflow.keras.layer import *
class Illumination_aware_Network:
    '''
    : [input] rgb image tensor
    : [output] weight c and t
    '''
    def illumination_mechanism(input_tensor_rgb,trainable=True):
        '''
        :[input] rgb tensor
        :[output] illuminate_output,w_n_weight,w_d_weight,w_rgb
        '''
        if K.image_dim_ordering() == 'tf':#bn_axis = 3
            bn_axis = 3
        else:
            bn_axis = 1
        # normalize the imput
        tf_resize_images = Lambda(lambda x: tf.image.resize_bilinear(x, [56, 56]))
        tf_div = Lambda(lambda x: x / 255)
        tf_original1 = Lambda(lambda x: x[:, :, :, 0] + 103.939)
        tf_original2 = Lambda(lambda x: x[:, :, :, 1] + 116.779)
        tf_original3 = Lambda(lambda x: x[:, :, :, 2] + 123.68)
        tf_expand_dims = Lambda(lambda x: tf.expand_dims(x, -1))

        img_input_rgb1 = tf_original1(input_tensor_rgb)
        img_input_rgb1 = tf_expand_dims(img_input_rgb1)
        img_input_rgb2 = tf_original2(input_tensor_rgb)
        img_input_rgb2 = tf_expand_dims(img_input_rgb2)
        img_input_rgb3 = tf_original3(input_tensor_rgb)
        img_input_rgb3 = tf_expand_dims(img_input_rgb3)
        img_input_rgb_pre = Concatenate()([img_input_rgb1, img_input_rgb2, img_input_rgb3])

        img_input_concat_resize = tf_resize_images(img_input_rgb_pre)
        img_input_concat_resize = tf_div(img_input_concat_resize)

        # the main of the IAN
        # predict the w_n,w_d
        img_input_concat_stage1 = Convolution2D(64, (3, 3), strides=(1, 1), name='illuminate_aware_stage1', padding='same',kernel_initializer='glorot_normal', trainable=trainable)(img_input_concat_resize)
        img_input_concat_stage1 = FixedBatchNormalization(axis=bn_axis, name='illuminate_aware_stage1_bn')(img_input_concat_stage1)
        img_input_concat_stage1 = Activation('relu')(img_input_concat_stage1)
        img_input_concat_stage1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(img_input_concat_stage1)

        img_input_concat_stage2 = Convolution2D(32, (3, 3), strides=(1, 1), name='illuminate_aware_stage2', padding='same',kernel_initializer='glorot_normal', trainable=trainable)(img_input_concat_stage1)
        img_input_concat_stage2 = FixedBatchNormalization(axis=bn_axis, name='illuminate_aware_stage2_bn')(img_input_concat_stage2)
        img_input_concat_stage2 = Activation('relu')(img_input_concat_stage2)
        img_input_concat_stage2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(img_input_concat_stage2)
        img_input_concat_stage2 = Flatten()(img_input_concat_stage2)

        img_input_concat_dense = Dense(units=128, activation='relu', name='illuminate_aware_dense1')(img_input_concat_stage2)
        img_input_concat_dense = Dropout(0.5)(img_input_concat_dense)
        img_input_concat_dense = Dense(units=64, activation='relu', name='illuminate_aware_dense2')(img_input_concat_dense)
        img_input_concat_dense = Dropout(0.5)(img_input_concat_dense)
        w_n = Dense(units=1, activation='relu', name='illuminate_aware_dense3')(img_input_concat_dense)
        w_d = Dense(units=1, activation='relu', name='illuminate_aware_dense4')(img_input_concat_dense)
        illuminate_output = Concatenate()([w_n, w_d])

        w_n_weight = Activation('sigmoid')(w_n)  # LWIR
        w_d_weight = Activation('sigmoid')(w_d)  # RGB

        # predict the w_absolute(|w|)
        img_input_concat_stage22 = Convolution2D(32, (3, 3), strides=(1, 1), name='illuminate_aware_stage22',padding='same', kernel_initializer='glorot_normal', trainable=trainable)(img_input_concat_stage1)
        img_input_concat_stage22 = FixedBatchNormalization(axis=bn_axis, name='illuminate_aware_stage22_bn')(img_input_concat_stage22)
        img_input_concat_stage22 = Activation('relu')(img_input_concat_stage22)
        img_input_concat_stage22 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(img_input_concat_stage22)
        img_input_concat_stage22 = Flatten()(img_input_concat_stage22)

        img_input_concat_dense_alf = Dense(units=128, activation='sigmoid', name='illuminate_aware_dense1_alf')(img_input_concat_stage22)
        img_input_concat_dense_alf = Dropout(0.5)(img_input_concat_dense_alf)
        img_input_concat_dense_alf = Dense(units=64, activation='sigmoid', name='illuminate_aware_dense2_alf')(img_input_concat_dense_alf)
        img_input_concat_dense_alf = Dropout(0.5)(img_input_concat_dense_alf)
        w_absolute = Dense(units=1, activation='sigmoid', name='illuminate_aware_dense3_alf')(img_input_concat_dense_alf)

        illuminate_aware_alf_value = Scale_bias(gamma_init=1.0, beta_init=0.0,name='illuminate_aware_alf_value_scale_bais')(w_absolute)

        tf_half_add = Lambda(lambda x: 0.5 + x)
        tf_sub = Lambda(lambda x: (x[0] - x[1])*0.5)
        # the final illumination weight
        w_n_illuminate = Activation('tanh')(w_n)  # LWIR
        w_d_illuminate = Activation('tanh')(w_d)  # RGB
        illuminate_rgb_positive = tf_sub([w_d_illuminate,w_n_illuminate])
        # illuminate_rgb_positive = tf_half_sub(w_n_illuminate)
        illuminate_aware_alf_pre = multiply([illuminate_rgb_positive, illuminate_aware_alf_value])
        w_rgb = tf_half_add(illuminate_aware_alf_pre)
        return illuminate_output,w_n_weight,w_d_weight,w_rgb


    #illumination Gate
    def Illumination_Gate(stage_rgb,stage_lwir,channel_num ,bn_axis,w_d_weight,w_n_weight,stage_name = 'stage3',trainable=True):
        stage_rgb = multiply([stage_rgb,w_d_weight])
        stage_lwir = multiply([stage_lwir,w_n_weight])
        stage_concat=Concatenate()([stage_rgb,stage_lwir])
        stage_concat = L2Normalization(gamma_init=10, name=stage_name+'_cat_bn_pre')(stage_concat)
        stage = Convolution2D(channel_num, (1, 1), strides=(1, 1),name=stage_name+'_concat_new',padding='same',kernel_initializer='glorot_normal', trainable=trainable)(stage_concat)
        stage = FixedBatchNormalization(axis=bn_axis, name=stage_name+'_cat_bn_new')(stage)
        stage = Activation('relu')(stage)
        return stage,stage_rgb,stage_lwir