
import __future__
import torch
from mvaggregate import MVAggregate
from torchvision.models.video import r3d_18, R3D_18_Weights, MC3_18_Weights, mc3_18
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, s3d, S3D_Weights
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights, mvit_v1_b, MViT_V1_B_Weights

from multidim_stacker_mod import MultiDimStacker


class MVNetwork(torch.nn.Module):

    def __init__(self, net_name='r2plus1d_18', agr_type='max', lifting_net=torch.nn.Sequential(), mv_dropout=0.2, 
                 freeze_backbone=False, skip_inter_layer=False, num_frames=-1):
        super().__init__()

        self.net_name = net_name
        self.agr_type = agr_type
        self.lifting_net = lifting_net
        
        self.feat_dim = 512

        if net_name == "r3d_18":
            weights_model = R3D_18_Weights.DEFAULT
            network = r3d_18(weights=weights_model)
        elif net_name == "s3d":
            weights_model = S3D_Weights.DEFAULT
            network = s3d(weights=weights_model)
            self.feat_dim = 400
        elif net_name == "mc3_18":
            weights_model = MC3_18_Weights.DEFAULT
            network = mc3_18(weights=weights_model)
        elif net_name == "r2plus1d_18":
            weights_model = R2Plus1D_18_Weights.DEFAULT
            network = r2plus1d_18(weights=weights_model)
        elif net_name == "mvit_v2_s":
            weights_model = MViT_V2_S_Weights.DEFAULT
            network = mvit_v2_s(weights=weights_model)
            self.feat_dim = 400
        elif(net_name=='multidim_stacker'):
            network = MultiDimStacker(model_name="tf_efficientnetv2_b0.in1k", num_3d_stack_proj=256, num_classes=15, drop_rate=mv_dropout, drop_path_rate=0.6) #num_classes is ignored, just don't put 0
            self.feat_dim = 1280
            #Load weights
            weights_path = './model_weights/ms_backbone_pretrained.pth'
            model_dict = torch.load(weights_path)
            
            network.load_state_dict(model_dict['no_ema_nn_state_dict'])
        elif(net_name=='multidim_stacker_color'):
            network = MultiDimStacker(model_name="tf_efficientnetv2_b0.in1k", num_frames=num_frames, num_classes=15, drop_rate=mv_dropout, drop_path_rate=0.6) #num_classes is ignored, just don't put 0
            self.feat_dim = 1280
            
        else:
            raise ValueError('Network does not exist.')
        
        print('Using {} as feature extractor.'.format(net_name))
                
        network.fc = torch.nn.Sequential()

        

        self.mvnetwork = MVAggregate(
            model=network,
            agr_type=self.agr_type, 
            feat_dim=self.feat_dim, 
            lifting_net=self.lifting_net,
            drop_rate=mv_dropout,
            freeze_backbone=freeze_backbone,
            skip_inter_layer=skip_inter_layer,
        )

    def forward(self, mvimages):
        return self.mvnetwork(mvimages)
