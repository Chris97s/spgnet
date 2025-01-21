import torch.nn as nn
import torch
from .ASM_Transformation.layer import HomogeneousShapeLayer
from .ASM_Transformation.networks import SingleShapeNetwork
import warnings
from torchvision.models import resnet18
warnings.filterwarnings("ignore",category=DeprecationWarning)
from .SGFC_block.SGFC import SGFC

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

class Decoder(nn.Module):
    def __init__(self,nb_filter = [64, 128, 256, 512,1024],num_classes=2,num_points = 128):
        super(Decoder, self).__init__()
        
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv3 = VGGBlock(nb_filter[4]+nb_filter[3], nb_filter[3], nb_filter[3])
        self.conv2 = VGGBlock(nb_filter[3]+nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv1 = VGGBlock(nb_filter[2]+nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv0 = VGGBlock(nb_filter[1]+nb_filter[0], nb_filter[0], nb_filter[0])

        
        #self.final_seg = nn.Conv2d(nb_filter[0]+1, num_classes, kernel_size=3,padding=1)
        self.final_seg = nn.Conv2d(nb_filter[0], num_classes, kernel_size=3,padding=1)
        
        

    def forward(self,segneck,features_encoder):#,revise2):    #ROI_feature size:bs*128,32,16,16
        # decoder_list = []
        
        segneck_up = self.up(segneck)
        # x3_att = self.att3(segneck_up,features_encoder[3])
        x3 = self.conv3(torch.cat([features_encoder[3], segneck_up], 1))
        #decoder_list.append(x3_1)
        
        x3_up = self.up(x3)
        # x2_att = self.att2(x3_up,features_encoder[2])
        x2 = self.conv2(torch.cat([features_encoder[2], x3_up], 1))
        #decoder_list.append(x2_2)
        
        x2_up = self.up(x2)
        # x1_att = self.att1(x2_up,features_encoder[1])
        x1 = self.conv1(torch.cat([features_encoder[1], x2_up], 1))
        #decoder_list.append(x1_3)
        
        x1_up = self.up(x1)
        # x0_att = self.att0(x1_up,features_encoder[0])
        x0 = self.conv0(torch.cat([features_encoder[0], x1_up], 1))
        #decoder_list.append(x1_3)
        
        #output_seg = self.final_seg(torch.concat([x0,revise2.unsqueeze(1)],dim=1))
        output_seg = self.final_seg(x0)
        return output_seg#,decoder_list

class SPGNet(nn.Module): 
    def __init__(self, in_channels=3, n_classes=2,img_size=256,shape_nclasses=3,n_evecs = 28,shape_prior = None,number_point = 0):
        super(SPGNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.shape_prior = shape_prior
        self.extrator_classification = resnet18(num_classes = shape_nclasses)
        self.shapenet = SingleShapeNetwork(                                              
                        HomogeneousShapeLayer, 
                        n_dims = 2,
                        use_cpp = False,
                        img_size=img_size,
                        in_channels = in_channels,
                        norm_type= 'instance',
                        feature_extractor= False,
                        n_evecs=n_evecs
                        )
        self.corse2fine = SGFC(num_point = number_point, d_model = 64,
                                    trainable = True, return_interm_layers = True,
                                    dilation = False, nhead = 8,
                                    feedforward_dim = 1024)
        self.decoder = Decoder(nb_filter=[64, 128, 256, 512,1024],num_classes=2,num_points = number_point)

    def forward(self,x):   #   shape(i): (1,1+num_component,num_pts,2)
        shape_class =torch.sigmoid(self.extrator_classification(x)) 
        extrator = shape_class.permute(1,0)
        shapes = self.shape_prior
        shapes = torch.matmul(shapes.double(),extrator.double()).permute(3,0,1,2)
        mean = shapes[:,0].float()       
        component = shapes[:,1:].float()  
        
        lmks,features_encoder,segneck = self.shapenet(x,mean,component)
        
        corse2fine_lmks = self.corse2fine(features_encoder,lmks)
        
        segmap = torch.sigmoid(self.decoder(segneck,features_encoder))
        
        return shape_class,lmks,corse2fine_lmks,segmap
