from utils import batch_tensor, unbatch_tensor
import torch
from torch import nn
import numpy as np


# Assuming your tensor is named 'tensor'
#MAX_FLOAT16 = torch.finfo(torch.float16).max  # Get the maximum value of float16

class WeightedAggregate(nn.Module):
    def __init__(self,  model, feat_dim, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        num_heads = 8
        self.feature_dim = feat_dim

        r1 = -1
        r2 = 1
        self.attention_weights = nn.Parameter((r1 - r2) * torch.rand(feat_dim, feat_dim) + r2)

        self.normReLu = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU()
        )        

        self.relu = nn.ReLU()
   


    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, N_frames, Height, Width
        #import pdb; pdb.set_trace()
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        '''
        foo = batch_tensor(mvimages, dim=1, squeeze=True) #torch.Size([4, 3, 16, 224, 224])
        foo2 = self.model(foo) #torch.Size([4, 400])
        foo3 = unbatch_tensor(foo2, B, dim=1, unsqueeze=True) #torch.Size([2, 2, 400])
        '''

        ##################### VIEW ATTENTION #####################

        # S = source length 
        # N = batch size
        # E = embedding dimension
        # L = target length

        aux = torch.matmul(aux, self.attention_weights)
        # Dimension S, E for two views (2,512)

        aux /= 2^4 #Prevent overflow for FP16. Not exhaustively tested, 2^4 seems to work fine.

        # Dimension N, S, E
        aux_t = aux.permute(0, 2, 1)

        prod = torch.bmm(aux, aux_t)
        

        relu_res = self.relu(prod)
        
        aux_sum = torch.sum(torch.reshape(relu_res, (B, V*V)).T, dim=0).unsqueeze(0)
        #import pdb; pdb.set_trace()
        final_attention_weights = torch.div(torch.reshape(relu_res, (B, V*V)).T, aux_sum.squeeze(0))
        final_attention_weights = final_attention_weights.T

        final_attention_weights = torch.reshape(final_attention_weights, (B, V, V))

        # final_attention_weights[:,0] = 0.0
        # row_sums = final_attention_weights.sum(dim=1, keepdim=True)
        # final_attention_weights /= row_sums


       # import pdb; pdb.set_trace()

        final_attention_weights = torch.sum(final_attention_weights, 1)

        

        output = torch.mul(aux.squeeze(), final_attention_weights.unsqueeze(-1))

        output = torch.sum(output, 1)

        return output.squeeze(), final_attention_weights


class ViewMaxAggregate(nn.Module):
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = torch.max(aux, dim=1)[0]
        return pooled_view.squeeze(), aux


class ViewAvgAggregate(nn.Module):
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = torch.mean(aux, dim=1)
        return pooled_view.squeeze(), aux
    

class MultidimStacking(nn.Module):
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width

        #Pick the view at pos 0
        mvimages = mvimages[:, 0, :, :, :, :]

        #mvimages = mvimages.view(B,C,D,H,W) #Read only one view
        mvimages = mvimages.permute(0,2,1,3,4) #B, 15, 1, 736, 1280

        #import pdb; pdb.set_trace()
        
        aux = self.lifting_net(self.model(mvimages))
        #aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = aux # torch.max(aux, dim=1)[0]
        return pooled_view.squeeze(), aux


class MVAggregate(nn.Module):
    def __init__(self,  model, agr_type="max", feat_dim=400, lifting_net=nn.Sequential(), drop_rate=0.1, featdim_intermediate_fract=1.0, 
                 freeze_backbone=False, skip_inter_layer=False):
        super().__init__()
        self.agr_type = agr_type
        self.skip_inter_layer = skip_inter_layer
        self.intermediate_feat_dim = 512

        # self.inter = nn.Sequential(
        #     nn.LayerNorm(feat_dim),
        #     nn.Linear(feat_dim, feat_dim), 
        #     nn.SiLU(), #Add nonlinearity, otherwise this is useless
        #     nn.Dropout(p=drop_rate),  
        #     nn.LayerNorm(feat_dim),                          
        #     nn.Linear(feat_dim, feat_dim),
        # )

        # self.fc_offence = nn.Sequential(
        #     nn.LayerNorm(feat_dim),
        #     nn.Linear(feat_dim, int(feat_dim*featdim_intermediate_fract)),
        #     nn.SiLU(),
        #     nn.Dropout(p=drop_rate),
        #     nn.LayerNorm(int(feat_dim*featdim_intermediate_fract)),
        #     nn.Linear(int(feat_dim*featdim_intermediate_fract), 4)
        # )

        # self.fc_action = nn.Sequential(
        #     nn.LayerNorm(feat_dim),
        #     nn.Linear(feat_dim, int(feat_dim*featdim_intermediate_fract)),
        #     nn.SiLU(),
        #     nn.Dropout(p=drop_rate),
        #     nn.LayerNorm(int(feat_dim*featdim_intermediate_fract)),
        #     nn.Linear(int(feat_dim*featdim_intermediate_fract), 8)
        # )

        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, self.intermediate_feat_dim), 
            nn.SiLU(), #Add nonlinearity, otherwise this is useless
            nn.Dropout(p=drop_rate),  
            #nn.LayerNorm(feat_dim),                          
            #nn.Linear(feat_dim, feat_dim),
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(self.intermediate_feat_dim),
            nn.Linear(self.intermediate_feat_dim, 256),
            nn.SiLU(),
            nn.Dropout(p=drop_rate),
            #nn.LayerNorm(256),
            nn.Linear(256, 4)
        )

        self.fc_action = nn.Sequential(
            nn.LayerNorm(self.intermediate_feat_dim),
            nn.Linear(self.intermediate_feat_dim, 256),
            nn.SiLU(),
            nn.Dropout(p=drop_rate),
            #nn.LayerNorm(256),
            nn.Linear(256, 8)
        )

        if(freeze_backbone is True):
            print('Freezing weights of backbone')
            # Freeze the weights of video_mae_encoder
            for param in model.parameters():
                #print('freezing {}'.format(param))
                param.requires_grad = False

        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAggregate(model=model, lifting_net=lifting_net)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAggregate(model=model, lifting_net=lifting_net)
        elif self.agr_type == "attention":
            self.aggregation_model = WeightedAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)
        elif self.agr_type == 'multidim_stacking':
            self.aggregation_model = MultidimStacking(model=model, lifting_net=lifting_net)
        else:
            raise ValueError('Pooling mode {} not implemented'.format(self.agr_type))
        
        

    def forward(self, mvimages):

        pooled_view, attention = self.aggregation_model(mvimages)
  
        if(self.skip_inter_layer):
            inter = pooled_view
        else:
            inter = self.inter(pooled_view)

        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)

        return pred_offence_severity, pred_action, attention
