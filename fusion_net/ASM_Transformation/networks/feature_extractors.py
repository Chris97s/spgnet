import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dRelu(torch.nn.Module):
    """
    Block holding one Conv2d and one ReLU layer
    """
    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        *args :
            positional arguments (passed to Conv2d)
        **kwargs :
            keyword arguments (passed to Conv2d)

        """
        super().__init__()
        self._conv = torch.nn.Conv2d(*args, **kwargs)
        self._relu = torch.nn.ReLU()

    def forward(self, input_batch):
        """
        Forward batch though layers

        Parameters
        ----------
        input_batch : :class:`torch.Tensor`
            input batch

        Returns
        -------
        :class:`torch.Tensor`
            result
        """
        # print(self._conv(input_batch).shape)
        return self._relu(self._conv(input_batch))

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Conv131_Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv3 = nn.Conv2d(middle_channels, out_channels, 1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out

class Conv3x3(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out
   
class Conv1x1(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out 

class Interaction(nn.Module):
    def __init__(self,C):
        super().__init__()
        
        self.fusion = Conv3x3(2*C,C)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(C, C//16),
            nn.ReLU(inplace=True),
            nn.Linear(C//16, C),
        )
        self.spatial_branch_conv = Conv131_Block(C,C//2,C)
        self.channel_branch_conv = Conv131_Block(C,C//2,C)
        
        
        self.res1 = Conv1x1(C,C)
        self.res2 = Conv1x1(C,C)
        
        
    def forward(self,input1,input2):
        fusion = self.fusion(torch.concat([input1,input2],dim=1))
        spatial_attention = torch.sigmoid(torch.mean(fusion,1).unsqueeze(1))   #bs,1,h,w
        channel_attention = self.mlp(F.avg_pool2d( fusion, (fusion.size(2), fusion.size(3)), stride=(fusion.size(2), fusion.size(3)))).unsqueeze(2).unsqueeze(3) #bs,c,1,1
        channel_attention = torch.sigmoid(channel_attention)
        attention_fusion = self.spatial_branch_conv(fusion) * spatial_attention + self.channel_branch_conv(fusion) * channel_attention
        out1 = self.res1(input1) + attention_fusion
        out2 = self.res2(input2) + attention_fusion
        return out1,out2

class DCM(nn.Module):
    def __init__(self, num_out_params, input_channels=3):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        
        ###E1
        self.shapenet_E1 = nn.Sequential(
            Conv2dRelu(input_channels, 64, (7, 1)),               #bs,64,250,256
            Conv2dRelu(64, 64, (1, 7))                            #bs,64,250,250
        )
        self.shapenet_E1_downsample = nn.Sequential(              #bs,128,122,122
            Conv2dRelu(64, 128, (7, 7), stride=2),
            nn.InstanceNorm2d(128)
        )
        self.feature_E1 = VGGBlock(input_channels,64,64)
        self.interaction1 = Interaction(64)
        
        ###E2
        self.shapenet_E2 = nn.Sequential(
            Conv2dRelu(128, 128, (7, 1)),                       #bs,128,116,122
            Conv2dRelu(128, 128, (1, 7))                        #bs,128,116,116
        )
        self.shapenet_E2_downsample = nn.Sequential(            #bs,256,55,55
            Conv2dRelu(128, 256, (7, 7), stride=2),
            nn.InstanceNorm2d(256)
        )
        self.feature_E2 = VGGBlock(64,128,128)
        self.interaction2 = Interaction(128)
        
        ###E3
        self.shapenet_E3 = nn.Sequential(
            Conv2dRelu(256, 256, (5, 1)),                       #bs,256,51,55
            Conv2dRelu(256, 256, (1, 5))                        #bs,256,51,51
        )
        self.shapenet_E3_downsample = nn.Sequential(            #bs,512,24,24
            Conv2dRelu(256, 512, (5, 5), stride=2),
            nn.InstanceNorm2d(512)
        )
        self.feature_E3 = VGGBlock(128,256,256)
        self.interaction3 = Interaction(256)
        
        ###E4
        self.shapenet_E4 = nn.Sequential(
            Conv2dRelu(512, 512, (5, 1)),                       #bs,512,20,24
            Conv2dRelu(512, 512, (1, 5))                        #bs,512,20,20
        )
        self.shapenet_E4_downsample = nn.Sequential(            #bs,256,8,8
            Conv2dRelu(512, 512, (5, 5), stride=2),
            nn.InstanceNorm2d(512)
        )
        self.feature_E4 = VGGBlock(256,512,512)
        self.interaction4 = Interaction(512)
        
        
        
        self.feature_E5 = VGGBlock(512,1024,1024)
        self.shapenet_E5 = nn.Sequential(
            Conv2dRelu(512, 256, (3, 1)),                       #bs,256,6,8
            Conv2dRelu(256, 256, (1, 3))                        #bs,256,6,6
        )
        
        
        self.shapenet_tail = nn.Sequential(
            Conv2dRelu(256, 256, (3, 1)),                      #bs,256,4,6
            Conv2dRelu(256, 256, (1, 3)),                      #bs,256,4,4
            Conv2dRelu(256, 128, (3, 1)),               
            Conv2dRelu(128, 128, (1, 3)),                      #bs,128,2,2
            nn.Conv2d(128, num_out_params,(2, 2))              #bs,num_out_params,1,1
        )
        
        
    def forward(self,x):
        feature = []
        #E1
        s1 = self.shapenet_E1(x)
        s1_size = s1.size(2)
        f1 = self.feature_E1(x)
        f1_size = f1.size(2)
        s1 = F.interpolate(s1, size=(f1_size,f1_size), mode='bilinear', align_corners=False)
        fusion_s1,fusion_f1 = self.interaction1(s1,f1)
        fusion_s1 = F.interpolate(fusion_s1, size=(s1_size,s1_size), mode='bilinear', align_corners=False)
        fusion_s1 = self.shapenet_E1_downsample(fusion_s1)
        feature.append(fusion_f1)
        fusion_f1 = self.pool(fusion_f1)
        
        
        
        #E2
        s2 = self.shapenet_E2(fusion_s1)
        s2_size = s2.size(2)
        f2 = self.feature_E2(fusion_f1)
        f2_size = f2.size(2)
        s2 = F.interpolate(s2, size=(f2_size,f2_size), mode='bilinear', align_corners=False)
        fusion_s2,fusion_f2 = self.interaction2(s2,f2)
        fusion_s2 = F.interpolate(fusion_s2, size=(s2_size,s2_size), mode='bilinear', align_corners=False)
        fusion_s2 = self.shapenet_E2_downsample(fusion_s2)
        feature.append(fusion_f2)
        fusion_f2 = self.pool(fusion_f2)
        
        
        #E3
        s3 = self.shapenet_E3(fusion_s2)
        s3_size = s3.size(2)
        f3 = self.feature_E3(fusion_f2)
        f3_size = f3.size(2)
        s3 = F.interpolate(s3, size=(f3_size,f3_size), mode='bilinear', align_corners=False)
        fusion_s3,fusion_f3 = self.interaction3(s3,f3)
        fusion_s3 = F.interpolate(fusion_s3, size=(s3_size,s3_size), mode='bilinear', align_corners=False)
        fusion_s3 = self.shapenet_E3_downsample(fusion_s3)
        feature.append(fusion_f3)
        fusion_f3 = self.pool(fusion_f3)
        
        
        #E4
        s4 = self.shapenet_E4(fusion_s3)
        s4_size = s4.size(2)
        f4 = self.feature_E4(fusion_f3)
        f4_size = f4.size(2)
        s4 = F.interpolate(s4, size=(f4_size,f4_size), mode='bilinear', align_corners=False)
        fusion_s4,fusion_f4 = self.interaction4(s4,f4)
        feature.append(fusion_f4)
        fusion_s4 = F.interpolate(fusion_s4, size=(s4_size,s4_size), mode='bilinear', align_corners=False)
        fusion_s4 = self.shapenet_E4_downsample(fusion_s4)    #bs,256,8,8
        fusion_f4 = self.pool(fusion_f4)                      #bs,512,16,16 
        
        
        #E5
        s5 = self.shapenet_E5(fusion_s4)
        f5 = self.feature_E5(fusion_f4)
        feature.append(f5)
        
        
        
        out_params = self.shapenet_tail(s5)
        return out_params,feature,f5
        
        
        