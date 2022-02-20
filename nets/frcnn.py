from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Input, Reshape,
                                     TimeDistributed)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal

from nets.resnet import ResNet50, classifier_layers
from nets.RoiPoolingConv import RoiPoolingConv

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
def get_classifier(conv_layers, input_rois, nb_classes=21, pooling_regions = 14):
    # num_rois, 38, 38, 1024 -> num_rois, 14, 14, 2048
    out_roi_pool = RoiPoolingConv(pooling_regions)([conv_layers, input_rois])
    
    # num_rois, 14, 14, 1024 -> num_rois, 1, 1, 2048
    out = classifier_layers(out_roi_pool)

    # num_rois, 1, 1, 1024 -> num_rois, 2048
    out = TimeDistributed(Flatten())(out)

    # num_rois, 1, 1, 1024 -> num_rois, nb_classes
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer=RandomNormal(stddev=0.02)), name='dense_class_{}'.format(nb_classes))(out)
    # num_rois, 1, 1, 1024 -> num_rois, 4 * (nb_classes-1)
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer=RandomNormal(stddev=0.02)), name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]

def get_model(config, num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    #----------------------------------------------------#
    #   Backbone
    #   假設輸入為600,600,3
    #   獲得一個38,38,1024的Feature Map (conv_layers)
    #----------------------------------------------------#
    conv_layers = ResNet50(inputs)
    #----------------------------------------------------#
    #   每個特徵點有9個anchor box
    #----------------------------------------------------#
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)

    #----------------------------------------------------#
    #   RPN
    #   將Feature Map (conv_layers)傳入RPN
    #   該網路結果會先對anchor box進行調整已獲得Region proposal box
    #----------------------------------------------------#
    rpn = get_rpn(conv_layers, num_anchors)
    model_rpn = Model(inputs, rpn)

    #----------------------------------------------------#
    #   將Feature Map (conv_layers)以及RPN傳入Head(classifier network)
    #   該網路結果會先對anchor box進行調整已獲得Region proposal box
    #   ROI Pooling → Head(classifier)
    #----------------------------------------------------#
    classifier = get_classifier(conv_layers, roi_input, num_classes, config.pooling_regions)

    model_all = Model([inputs, roi_input], rpn + classifier)
    return model_rpn, model_all

def get_predict_model(config, num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None,None,1024))
    #----------------------------------------------------#
    #   假設輸入為600,600,3
    #   獲得一個38,38,1024的Feature Map (conv_layers)
    #----------------------------------------------------#
    conv_layers = ResNet50(inputs)
    #----------------------------------------------------#
    #   每個特徵點有9個anchor box
    #----------------------------------------------------#
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)

    #----------------------------------------------------#
    #   將Feature Map (conv_layers)傳入RPN
    #   該網路結果會先對anchor box進行調整已獲得Region proposal box
    #----------------------------------------------------#
    rpn = get_rpn(conv_layers, num_anchors)
    model_rpn = Model(inputs, rpn + [conv_layers])

    #----------------------------------------------------#
    #   將Feature Map (conv_layers)以及RPN傳入Head(classifier network)
    #   該網路結果會先對anchor box進行調整已獲得Region proposal box
    #   ROI Pooling → Head(classifier)
    #----------------------------------------------------#
    classifier = get_classifier(feature_map_input, roi_input, num_classes, config.pooling_regions)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)
    return model_rpn, model_classifier_only
