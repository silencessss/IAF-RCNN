class Config:
    def __init__(self):
        self.anchor_box_scales = [128, 256, 512]
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
        self.rpn_stride = 16
        
        self.num_rois = 128
        #------------------------------------------------------#
        #   用於預測、用於訓練的Region proposal box數量
        #------------------------------------------------------#
        self.num_RPN_predict_pre = 300
        self.num_RPN_train_pre = 600

        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7
        
        #-------------------------------------------------------------------------------------#
        #   與真實框的IoU在classifier_min_overlap到classifier_max_overlap之間為負樣本
        #   與真實框的IoU大於classifier_max_overlap之間為正樣本
        #   由於訓練多了batch，如果將classifier_min_overlap設置為0.1，則可能存在無負樣本的情況
        #   將classifier_min_overlap下調為0，從而實現帶有batch的訓練
        #-------------------------------------------------------------------------------------#
        self.classifier_min_overlap = 0
        self.classifier_max_overlap = 0.5
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        self.pooling_regions = 14