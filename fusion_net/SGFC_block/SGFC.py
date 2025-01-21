import torch
import torch.nn as nn
import numpy as np
from .interpolation import interpolation_layer
from .get_roi import get_roi
from .Non_local import TransformerEncoder

class SGFC(nn.Module):
    def __init__(self, num_point, d_model, trainable,
                 return_interm_layers, dilation, nhead, feedforward_dim):
        super(SGFC, self).__init__()
        
        # Initialize hyperparameters
        self.num_point = num_point
        self.d_model = d_model
        self.trainable = trainable
        self.return_interm_layers = return_interm_layers
        self.dilation = dilation
        self.nhead = nhead
        self.feedforward_dim = feedforward_dim
        self.Sample_num = 16  # Sample points per ROI
        
        # ROI creators for different scales (feature map sizes)
        self.ROI_1 = get_roi(num_points=self.Sample_num, half_length=8.0, img_size=64)
        self.ROI_2 = get_roi(num_points=self.Sample_num, half_length=8.0, img_size=128)
        self.ROI_3 = get_roi(num_points=self.Sample_num, half_length=8.0, img_size=256)
        
        # Channel reduction layers for different scales
        self.Channel_reduction_1 = nn.Conv2d(256, d_model, kernel_size=3, padding=1)   
        self.Channel_reduction_2 = nn.Conv2d(128, d_model, kernel_size=3, padding=1)
        self.Channel_reduction_3 = nn.Conv2d(64, d_model, kernel_size=3, padding=1)

        # Interpolation layer
        self.interpolation = interpolation_layer()
        
        # Feature extractors
        self.feature_extractor_1 = nn.Conv2d(d_model, d_model, kernel_size=self.Sample_num, bias=False)
        self.feature_extractor_2 = nn.Conv2d(d_model, d_model, kernel_size=self.Sample_num, bias=False)
        self.feature_extractor_3 = nn.Conv2d(d_model, d_model, kernel_size=self.Sample_num, bias=False)
        
        # Non-local attention layers
        self.NLattention_1 = TransformerEncoder(d_model, num_heads=8, num_layers=8, num_points=self.num_point)
        self.NLattention_2 = TransformerEncoder(d_model, num_heads=8, num_layers=8, num_points=self.num_point)
        self.NLattention_3 = TransformerEncoder(d_model, num_heads=8, num_layers=8, num_points=self.num_point)
        
        # Output layers for each stage
        self.out_layer_1 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 2),
        )
        self.out_layer_2 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 2),
        )        
        self.out_layer_3 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 2),
        )

        # Location refinement layers
        self.LocRefine_1 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=self.Sample_num, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        self.LocRefine_fc_1 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 2),
        )
        
        self.LocRefine_2 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=self.Sample_num, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        self.LocRefine_fc_2 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 2),
        )
        
        self.LocRefine_3 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=self.Sample_num, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        self.LocRefine_fc_3 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, features, initial_landmarks):
        bs = features[0].size(0)

        output_list = []
        
        # Stage 1
        feature_map = features[2]   # bs,256,64,64
        feature_map = self.Channel_reduction_1(feature_map)  # bs,32,64,64
        
        ROI_anchor_1, bbox_size_1, start_anchor_1 = self.ROI_1(initial_landmarks)   
        
        ROI_anchor_1 = ROI_anchor_1.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_1 = self.interpolation(feature_map, ROI_anchor_1.detach()).view(bs, self.num_point, self.Sample_num,  
                                                                            self.Sample_num, self.d_model)
        ROI_feature_1 = ROI_feature_1.view(bs * self.num_point, self.Sample_num, self.Sample_num, self.d_model).permute(0, 3, 2, 1)

        offset_1 = self.LocRefine_1(ROI_feature_1)  # Local feature refinement
        offset_1 = offset_1.view(bs, self.num_point, -1)      
        offset_1 = self.LocRefine_fc_1(offset_1)            
        
        Nonlocal_ROI_feature_1 = self.feature_extractor_1(ROI_feature_1).view(bs, self.num_point, self.d_model)  
        offset_2 = self.out_layer_1(self.NLattention_1(Nonlocal_ROI_feature_1))
        
        offset = offset_1 + offset_2
        landmarks_1 = start_anchor_1 + bbox_size_1 * offset
        output_list.append(landmarks_1)

        # Stage 2   
        feature_map = features[1]   # bs,128,128,128
        feature_map = self.Channel_reduction_2(feature_map)  # bs,32,128,128
        
        ROI_anchor_2, bbox_size_2, start_anchor_2 = self.ROI_2(landmarks_1)   
        
        ROI_anchor_2 = ROI_anchor_2.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_2 = self.interpolation(feature_map, ROI_anchor_2.detach()).view(bs, self.num_point, self.Sample_num,  
                                                                            self.Sample_num, self.d_model)
        ROI_feature_2 = ROI_feature_2.view(bs * self.num_point, self.Sample_num, self.Sample_num, self.d_model).permute(0, 3, 2, 1)

        offset_1 = self.LocRefine_2(ROI_feature_2)
        offset_1 = offset_1.view(bs, self.num_point, -1)
        offset_1 = self.LocRefine_fc_2(offset_1)
        
        Nonlocal_ROI_feature_2 = self.feature_extractor_2(ROI_feature_2).view(bs, self.num_point, self.d_model)  
        offset_2 = self.out_layer_2(self.NLattention_2(Nonlocal_ROI_feature_2))
        
        offset = offset_1 + offset_2
        landmarks_2 = start_anchor_2 + bbox_size_2 * offset
        output_list.append(landmarks_2)
        
        # Stage 3   
        feature_map = features[0]   # bs,64,256,256
        feature_map = self.Channel_reduction_3(feature_map)  # bs,32,256,256
        
        ROI_anchor_3, bbox_size_3, start_anchor_3 = self.ROI_3(landmarks_2)   
        
        ROI_anchor_3 = ROI_anchor_3.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_3 = self.interpolation(feature_map, ROI_anchor_3.detach()).view(bs, self.num_point, self.Sample_num,  
                                                                            self.Sample_num, self.d_model)
        ROI_feature_3 = ROI_feature_3.view(bs * self.num_point, self.Sample_num, self.Sample_num, self.d_model).permute(0, 3, 2, 1)

        offset_1 = self.LocRefine_3(ROI_feature_3)
        offset_1 = offset_1.view(bs, self.num_point, -1)
        offset_1 = self.LocRefine_fc_3(offset_1)
        
        Nonlocal_ROI_feature_3 = self.feature_extractor_3(ROI_feature_3).view(bs, self.num_point, self.d_model)  
        offset_2 = self.out_layer_3(self.NLattention_3(Nonlocal_ROI_feature_3))
        
        offset = offset_1 + offset_2
        landmarks_3 = start_anchor_3 + bbox_size_3 * offset
        output_list.append(landmarks_3)

        return output_list
