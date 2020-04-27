import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from nets import * 



class DecodeEdge(nn.Module):
    def __init__(self, D, output_feat, otherstuff=None):
        super( DecodeEdge, self).__init__()

        self.l1 = nn.Linear(D, D)
        self.l2 = nn.Linear(D, D)
        self.l3 = nn.Linear(D, D)
        self.l4 = nn.Linear(D, output_feat)
        
    def forward(self, v):
        v1 = F.relu(self.l1(v))

        
        e = F.relu(v1.unsqueeze(1) +  v1.unsqueeze(2))
        
        e  = F.relu(self.l2(e))
        e_v = torch.max(e, dim=2)[0]
        e_v = torch.relu(self.l3(e_v))
        e2 = F.relu(e_v.unsqueeze(1) + e_v.unsqueeze(2))
        
        out  = self.l4(e + e2 )
        return out 


class Decode2(nn.Module):
    def __init__(self, D, output_feat, otherstuff=None):
        super( Decode2, self).__init__()

        self.l1_sum = nn.Linear(D, D)
        self.l1_prod = nn.Linear(D, D)
        self.l2 = nn.Linear(D, D)
        self.l3 = nn.Linear(D, D)
        self.l4 = nn.Linear(D, output_feat)
        self.norm = nn.LayerNorm(D)

    def forward(self, v):
        v1 = F.relu(self.l1_sum(v))

        v2 = F.relu(self.l1_prod(v))

        
        e = v1.unsqueeze(1) +  v1.unsqueeze(2)
        e = e + v2.unsqueeze(1) * v2.unsqueeze(2)
        
        e  = F.relu(self.l2(e))
        e_v = torch.max(e, dim=2)[0]
        e_v = torch.relu(self.l3(e_v))

        e2 = F.relu(e_v.unsqueeze(1) + e_v.unsqueeze(2))
        e5 = e + e2 
        e5_norm = self.norm(e5)
        out  = self.l4(e5_norm)
        return out 



class Decode3(nn.Module):
    def __init__(self, D, output_feat, out_transform=None):
        super( Decode3, self).__init__()

        self.l1_sum = nn.Linear(D, D)
        self.l1_prod = nn.Linear(D, D)
        self.l2 = nn.Linear(D, D)
        self.l3 = nn.Linear(D, D)
        self.l4 = nn.Linear(D, D)
        self.l5 = nn.Linear(D, output_feat)
        self.norm = nn.LayerNorm(D)
        self.out_transform = out_transform

    def forward(self, v):
        e_in = v.unsqueeze(1) *  v.unsqueeze(2)
        
        e  = F.relu(self.l2(e_in))

        e_v = torch.max(e, dim=2)[0]
        e_v = torch.relu(self.l3(e_v))

        e2 = F.relu(e_v.unsqueeze(1) + e_v.unsqueeze(2))
        e5 = e + e2 + e_in
        e5_norm = self.norm(e5)
        e6  = F.leaky_relu(self.l4(e5_norm))
        out = self.l5(e6)
        if self.out_transform == 'sigmoid':
            out = torch.sigmoid(out)
        elif self.out_transform == 'relu':
            out = F.relu(out)
            
        return out 



class Decode4(nn.Module):
    def __init__(self, D, output_feat,
                 input_dropout_p=0.0,
                 out_transform=None):
        super( Decode4, self).__init__()

        self.l1_sum = nn.Linear(D, D)
        self.l1_prod = nn.Linear(D, D)
        self.l2 = nn.Linear(D, D)
        self.l3 = nn.Linear(D, D)
        self.l4 = nn.Linear(D, D)
        self.l5 = nn.Linear(D, output_feat)
        self.norm = nn.LayerNorm(D)
        self.out_transform = out_transform

        self.input_dropout_p = input_dropout_p
        if input_dropout_p > 0:
            self.input_dropout = nn.Dropout(p=input_dropout_p)

    def forward(self, v):

        if self.input_dropout_p  > 0.0:
            v = self.input_dropout(v)
            
        e_in = v.unsqueeze(1) *  v.unsqueeze(2)
        
        e  = F.relu(self.l2(e_in))

        e_v = torch.max(e, dim=2)[0]
        e_v = torch.relu(self.l3(e_v))

        e2 = F.relu(e_v.unsqueeze(1) + e_v.unsqueeze(2))
        e5 = e + e2 + e_in
        e5_norm = self.norm(e5)
        e6  = F.leaky_relu(self.l4(e5_norm))
        out = self.l5(e6)
        if self.out_transform == 'sigmoid':
            out = torch.sigmoid(out)
        elif self.out_transform == 'relu':
            out = F.relu(out)
            
        return out 


class Decode5(nn.Module):
    def __init__(self, D, output_feat,
                 input_dropout_p=0.0,
                 out_transform=None):
        super( Decode5, self).__init__()

        self.edge_l = nn.Linear(D, D)

        self.out_transform = out_transform
        self.out_l = nn.Linear(D, output_feat)

        self.input_dropout_p = input_dropout_p
        if input_dropout_p > 0:
            self.input_dropout = nn.Dropout(p=input_dropout_p)

    def forward(self, v):

        if self.input_dropout_p  > 0.0:
            v = self.input_dropout(v)
            
        e_in = v.unsqueeze(1)  +   v.unsqueeze(2)
        
        e  = self.edge_l(F.leaky_relu(e_in))
        out = self.out_l(F.leaky_relu(e))
        
        if self.out_transform == 'sigmoid':
            out = torch.sigmoid(out)
        elif self.out_transform == 'relu':
            out = F.relu(out)
        return out 
    


class SemiNet1(nn.Module):
    def __init__(self, g_feature_n, g_feature_out_n=None, 
                 int_d = None, layer_n = None, 
                 mixture_n = 5, 
                 resnet=True, 
                 gml_class = 'GraphMatLayers',
                 gml_config = {}, 
                 decode_class = 'DecodeEdge', 
                 decode_config = {},
                 init_noise=1e-5, agg_func=None, GS=1, OUT_DIM=1, 
                 input_batchnorm=False, out_std= False, 
                 resnet_out = False, resnet_blocks = (3, ), 
                 resnet_d = 128, 
                 batchnorm=False, 
                 out_std_exp = False, 
                 force_lin_init=False, 
                 use_random_subsets=True, 
):
        
        """

        
        """
        super( SemiNet1, self).__init__()

        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n

        self.gml = eval(gml_class)(g_feature_n, g_feature_out_n, 
                                   resnet=resnet, noise=init_noise,
                                   agg_func=parse_agg_func(agg_func), 
                                   GS=GS,
                                   **gml_config)

        if input_batchnorm:
            self.input_batchnorm = MaskedBatchNorm1d(g_feature_n)
        else:
            self.input_batchnorm = None

        self.resnet_out = resnet_out 
        if not resnet_out:
            self.mix_out = nn.ModuleList([nn.Linear(g_feature_out_n[-1], OUT_DIM) for _ in range(mixture_n)])
        else:
            self.mix_out = nn.ModuleList([ResNetRegression(g_feature_out_n[-1], 
                                                           block_sizes = resnet_blocks, 
                                                           INT_D = resnet_d, 
                                                           FINAL_D=resnet_d, 
                                                           OUT_DIM=OUT_DIM) for _ in range(mixture_n)])

        self.out_std = out_std
        self.out_std_exp = False

        self.use_random_subsets = use_random_subsets

        self.decode_edge = eval(decode_class)(g_feature_out_n[-1], 4, 
                                              **decode_config)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


    def forward(self, adj, vect_feat, input_mask, input_idx, adj_oh, 
                return_g_features = False, **kwargs):
        G = adj
        for var in [adj, vect_feat, input_mask, input_idx, adj_oh]:
            assert not torch.isnan(var).any()
        BATCH_N, MAX_N, F_N = vect_feat.shape

        if self.input_batchnorm is not None:
            vect_feat_flat = vect_feat.reshape(BATCH_N*MAX_N, F_N)
            input_mask_flat = input_mask.reshape(BATCH_N * MAX_N)
            vect_feat_out_flat = self.input_batchnorm(vect_feat_flat, input_mask_flat)
            vect_feat = vect_feat_out_flat.reshape(BATCH_N, MAX_N, F_N)
        
        G_features = self.gml(G, vect_feat, input_mask)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])
        
        if self.resnet_out:
            x_1 = [m(g_squeeze_flat).reshape(BATCH_N, MAX_N, -1) for m in self.mix_out]
        else:
            x_1 = [m(g_squeeze) for m in self.mix_out]

        x_1 = torch.stack(x_1)
        assert not torch.isnan(x_1).any()
        if self.training:
            x_zeros = np.zeros(x_1.shape)
            if self.use_random_subsets:
                rand_ints = np.random.randint(x_1.shape[0], size=BATCH_N)
            else:
                rand_ints = (input_idx % len(self.mix_out)).cpu().numpy()
            #print(rand_ints)
            for i, v in enumerate(rand_ints):
                x_zeros[v, i, :, :] = 1
            x_1_sub = torch.Tensor(x_zeros).to(x_1.device) * x_1
            x_1_sub = x_1_sub.sum(dim=0)
            assert not torch.isnan(x_1_sub).any()

        else:
            x_1_sub = x_1.mean(dim=0)
        # #print(x_1.shape)
        # idx = torch.randint(high=x_1.shape[0], 
        #                     size=(BATCH_N, )).to(G.device)
        # #print("idx=", idx)
        # x_1_sub = torch.stack([x_1[v, v_i] for v_i, v in enumerate(idx)])
        if len(self.mix_out) == 1:
            std = torch.mean(torch.ones_like(x_1), dim=0)
        else:
            std = torch.sqrt(torch.var(x_1, dim=0) + 1e-5)

            #print("numpy_std=", np.std(x_1.detach().cpu().numpy()))

        #kjlkjalds
        x_1 = x_1_sub

        decode_edge = self.decode_edge(g_squeeze)
        G_perm = adj_oh.permute(0, 2, 3, 1)
        assert G_perm.shape == decode_edge.shape

        return {'mu' : x_1, 'std' : std, 'g_in' : G_perm, 
                'g_decode' : decode_edge}


class SemiNet2(nn.Module):
    def __init__(self,
                 g_feature_n,
                 GS, 
                 encode_class = 'GraphVertConfigBootstrap',
                 encode_config = {}, 
                 decode_class = 'DecodeEdge', 
                 decode_config = {},
                 **kwargs,
    ):
        
        """

        
        """
        super( SemiNet2, self).__init__()

        for k, v in kwargs.items():
            print(f"arg {k}={v} ")
            
        for k, v in encode_config.items():
            print(f"encode_config arg {k}={v} ")
            
        
        self.encode = eval(encode_class)(g_feature_n=g_feature_n,
                                         GS = GS, 
                                         **encode_config)

        self.decode_edge = eval(decode_class)(**decode_config)

    def forward(self, adj, vect_feat, input_mask, input_idx, adj_oh, 
                return_g_features = False, **kwargs):

        encode = self.encode(adj, vect_feat, input_mask, input_idx, adj_oh, 
                             also_return_g_features = True)

        mu = encode['mu']
        std = encode['std']
        g_features = encode['g_features']
                             

        decode_edge = self.decode_edge(g_features)
        G_perm = adj_oh.permute(0, 2, 3, 1)
        #print("g_features.shape=", g_features.shape,
        #      "G_perm.shape=", G_perm.shape,
        #      "decode_edge.shape=", decode_edge.shape)
        
        #assert G_perm.shape == decode_edge.shape

        return {'mu' : mu, 'std' : std, 'g_in' : G_perm, 
                'g_decode' : decode_edge}



class ReconLoss(nn.Module):
    def __init__(self, pred_norm='l2', pred_scale=1.0, 
                 pred_loss_weight = 1.0, 
                 recon_loss_weight = 1.0, loss_name=None, 
                 recon_loss_name = 'nn.BCEWithLogitsLoss', 
                 recon_loss_config = {}):
        super(ReconLoss, self).__init__()

        self.pred_loss = NoUncertainLoss(pred_norm, pred_scale)
        self.recon_loss = eval(recon_loss_name)(reduction='none', 
                                               **recon_loss_config)
        self.pred_loss_weight = pred_loss_weight
        self.recon_loss_weight = recon_loss_weight


    def __call__(self, pred, y, pred_mask, input_mask):
        assert not torch.isnan(y).any()
        assert not torch.isnan(pred_mask).any()
        for k, v in pred.items():
            if  torch.isnan(v).any():
                raise Exception(f"k={k}") 

        if torch.sum(pred_mask) > 0:
            l_pred = self.pred_loss(pred, y, pred_mask, input_mask)
            assert not torch.isnan(l_pred).any()
        else:
            l_pred = torch.tensor(0.0).to(y.device)


        BATCH_N = y.shape[0]
        
        input_mask_2d = input_mask.unsqueeze(1) * input_mask.unsqueeze(-1)

        g_in = pred['g_in'] #  .reshape(BATCH_N, -1)
        MAX_N = g_in.shape[1]

        g_decode = pred['g_decode']#.reshape(BATCH_N, -1)
        l_recon = self.recon_loss(g_decode, g_in)
        assert not torch.isnan(l_recon).any()

        l_recon = l_recon[input_mask_2d.unsqueeze(-1).expand(BATCH_N, MAX_N, MAX_N, 4)>0].mean()
        #l_recon = l_recon.mean()
                              
        loss = self.pred_loss_weight* l_pred + self.recon_loss_weight * l_recon
        return {'loss' : loss, 
                'loss_recon' : l_recon, 
                'loss_pred' : l_pred}
