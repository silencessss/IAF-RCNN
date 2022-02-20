import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Input, Reshape,
                                     TimeDistributed)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import plot_model
from nets.resnet import ResNet50, classifier_layers
from nets.RoiPoolingConv import RoiPoolingConv
from utils.config import Config
config = Config()
#----------------------------------------------------#
#   Region Proposal Network (RPN)
#   該網路結果會先對anchor box進行調整已獲得Region proposal box
#----------------------------------------------------#
def get_rpn(conv_layers, num_anchors):
    '''
    : input: conv_layer. (feature map)
    : output: class, regression. (proposal)
    '''
    #----------------------------------------------------#
    #   利用一個 512通道的3x3Conv.進行特徵提取
    #----------------------------------------------------#
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=RandomNormal(stddev=0.02), name='rpn_conv1')(conv_layers)

    #----------------------------------------------------#
    #   利用一個1x1 Conv.调整通道数，獲得預測結果
    #----------------------------------------------------#
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer=RandomNormal(stddev=0.02), name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer=RandomNormal(stddev=0.02), name='rpn_out_regress')(x)
    
    x_class = Reshape((-1,1),name="classification")(x_class)
    x_regr = Reshape((-1,4),name="regression")(x_regr)
    return [x_class, x_regr]
#----------------------------------------------------#
#   將Feature Map (conv_layers)以及RPN傳入Head(classifier network)
#   該網路結果會先對anchor box進行調整已獲得Region proposal box
#   ROI Pooling → Head(classifier)
#----------------------------------------------------#
def get_classifier(conv_layers, input_rois, nb_classes=21, pooling_regions = 14, name='rgb'):
    # num_rois, 38, 38, 1024 -> num_rois, 14, 14, 2048
    out_roi_pool = RoiPoolingConv(pooling_regions)([conv_layers, input_rois])
    
    # num_rois, 14, 14, 1024 -> num_rois, 1, 1, 2048
    out = classifier_layers(out_roi_pool,name)

    # num_rois, 1, 1, 1024 -> num_rois, 2048
    out = TimeDistributed(Flatten())(out)

    # num_rois, 1, 1, 1024 -> num_rois, nb_classes
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer=RandomNormal(stddev=0.02)), name=name+'dense_class_{}'.format(nb_classes))(out)
    # num_rois, 1, 1, 1024 -> num_rois, 4 * (nb_classes-1)
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer=RandomNormal(stddev=0.02)), name=name+'dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]


def get_model(config, num_classes):
    # input
    inputs_rgb = Input(shape=(None,None,3))
    inputs_thermal = Input(shape=(None,None,3))
    inputs = [inputs_rgb,inputs_thermal]
    roi_input = Input(shape=(None,4))
    #----------------------------------------------------#
    #   Backbone
    #----------------------------------------------------#
    conv_layers_rgb = ResNet50(inputs_rgb,name='rgb_')
    print(conv_layers_rgb.shape)
    conv_layers_thermal = ResNet50(inputs_thermal,name='thermal_')
    print(conv_layers_thermal.shape)
    conv_layers_concate = tf.concat([conv_layers_rgb,conv_layers_thermal],1)
    print(conv_layers_concate.shape)
    #----------------------------------------------------#
    #   anchor box
    #----------------------------------------------------#
    num_anchors = len(config.anchor_box_scales)*len(config.anchor_box_ratios)
    #----------------------------------------------------#
    #   Region proposal network (RPN)
    #----------------------------------------------------#
    rpn = get_rpn(conv_layers_concate, num_anchors)
    #model_rpn = Model(inputs,rpn)
    #----------------------------------------------------#
    #   RoI pooling and model Head
    #----------------------------------------------------#
    classifier_rgb_class,classifier_rgb_regr = get_classifier(conv_layers_rgb,roi_input,num_classes,config.pooling_regions, name='rgb_')
    classifier_thermal_class,classifier_thermal_regr = get_classifier(conv_layers_thermal,roi_input,num_classes,config.pooling_regions, name='thermal_')
    print('###')
    classifier_fusion_class = tf.add(classifier_rgb_class,classifier_thermal_class)
    classifier_fusion_regr = tf.add(classifier_rgb_regr,classifier_thermal_regr)
    print(classifier_fusion_class)
    print(classifier_fusion_regr)

    model_all = Model([inputs,roi_input],rpn+[classifier_fusion_class,classifier_fusion_regr])
    return model_all

def main():
    model = get_model(config,4)
    #model.summary()
    #----------------------------------------------------------#
    # PLOT Model Archicture
    #----------------------------------------------------------#
    # https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model
    # [rankdir]
    # rankdir argument passed to PyDot, a string specifying the format of the plot: 
    # 'TB' creates a vertical plot; 'LR' creates a horizontal plot.
    #----------------------------------------------------------#
    print('[INFO] Draw model..')
    plot_model(
        model,
        to_file='model_output.jpg',
        show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )

main()
