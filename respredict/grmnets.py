import nets
import torch
from torch import nn
import torch.nn.functional as F


class GraphMatResLayer(nn.Module):
    def __init__(self, C, P , GS=1,  res_depth=4, res_int_d = 128, 
                 noise=1e-6, agg_func=None):
        """
        Pairwise layer -- takes a N x M x M x C matrix
        and turns it into a N x M x M x P matrix after
        multiplying with a graph matrix N x M x M
        
        if GS != 1 then there will be a per-graph-channel 
        linear layer
        """
        super(GraphMatResLayer, self).__init__()
        
        self.GS = GS
        self.noise=noise

        self.linlayers = nn.ModuleList()
        
        for ll in range(GS):
            l = nets.ResNet(input_dim=C, hidden_dim=res_int_d, depth=res_depth,
                            output_dim =P, init_std=noise)
            self.linlayers.append(l)
            
            
        #self.r = nn.PReLU()
        self.r = nn.ReLU()
        self.agg_func = agg_func
 
    def forward(self, G, x):
        def apply_ll(i, x):
            x_flat = x.reshape(-1, x.shape[-1])
            y = self.linlayers[i](x_flat)
            y = y.reshape(x.shape[0], x.shape[1], -1)
            return y
        multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)])
        # this is per-batch-element
        xout = torch.stack([torch.matmul(G[i], multi_x[:, i]) for i in range(x.shape[0])])
        
        x = self.r(xout)
        if self.agg_func is not None:
            x = self.agg_func(x, dim=1)
        return x
    

class GraphMatResLayers(nn.Module):
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 g_res_depth = 2, 
                 g_res_int_d = 128, 
                 noise=1e-5, agg_func=None, dropout=0.0):
        super(GraphMatResLayers, self).__init__()
        
        self.gl = nn.ModuleList()
        self.resnet = resnet

        for li in range(len(output_features_n)):
            if li == 0:
                gl = GraphMatResLayer(input_feature_n, output_features_n[0],
                                      noise=noise, agg_func=agg_func, GS=GS, 
                                      res_depth= g_res_depth, 
                                      res_int_d = g_res_int_d)
            else:
                gl = GraphMatResLayer(output_features_n[li-1], 
                                      output_features_n[li], 
                                      noise=noise, agg_func=agg_func, GS=GS, 
                                      res_depth= g_res_depth, 
                                      res_int_d = g_res_int_d)
            
            self.gl.append(gl)
    def forward(self, G, x):
        for gl in self.gl:
            x2 = gl(G, x)
            if self.resnet:
                if x.shape == x2.shape:
                    x3 = x2 + x
                else:
                    x3 = x2
            else:
                x3 = x2
            x = x3
        

        return x


class GraphVertGRLModel(nn.Module):
    def __init__(self, g_feature_n, g_feature_out_n=None, 
                 int_d = None, layer_n = None, 
                 resnet=True, 
                 grl_res_depth=2, 
                 grl_res_int_d = 128, 
                 init_noise=1e-5, agg_func=None, GS=1, OUT_DIM=1, 
                 batch_norm=False, out_std= False, 
                 resnet_out = False, resnet_blocks = (3, ), 
                 resnet_d = 128, 
                 graph_dropout=0.0, 
                 force_lin_init=False):
        
        """

        
        """
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n


        super(GraphVertGRLModel, self).__init__()
        self.gml = GraphMatResLayers(g_feature_n, g_feature_out_n, 
                                     resnet=resnet, noise=init_noise, agg_func=nets.parse_agg_func(agg_func), 
                                     g_res_depth = grl_res_depth, 
                                     g_res_int_d = grl_res_int_d, 
                                     GS=GS, dropout=graph_dropout)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(g_feature_n)
        else:
            self.batch_norm = None

        self.resnet_out = resnet_out
        if not resnet_out:
            self.lin_out = nn.Linear(g_feature_out_n[-1], OUT_DIM)
        else:
            self.lin_out = nets.ResNetRegression(g_feature_out_n[-1], 
                                                block_sizes = resnet_blocks, 
                                                INT_D = resnet_d, 
                                                FINAL_D=resnet_d, 
                                                OUT_DIM=OUT_DIM)

        self.out_std = out_std

        if out_std:
            self.lin_out_std1 = nn.Linear(g_feature_out_n[-1], 128)
            self.lin_out_std2 = nn.Linear(128, OUT_DIM)

            # self.lin_out_std = ResNetRegression(g_feature_out_n[-1], 
            #                                     block_sizes = resnet_blocks, 
            #                                     INT_D = resnet_d, 
            #                                     FINAL_D=resnet_d, 
            #                                     OUT_DIM=OUT_DIM)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, init_noise)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, args):
        (G, x_G) = args
        
        BATCH_N, MAX_N, F_N = x_G.shape

        if self.batch_norm is not None:
            x_G_flat = x_G.reshape(BATCH_N*MAX_N, F_N)
            x_G_out_flat = self.batch_norm(x_G_flat)
            x_G = x_G_out_flat.reshape(BATCH_N, MAX_N, F_N)
        
        G_features = self.gml(G, x_G)

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])
        
        if self.resnet_out:
            x_1 = self.lin_out(g_squeeze_flat).reshape(BATCH_N, MAX_N, -1)
        else:
            x_1 = self.lin_out(g_squeeze)
        if self.out_std:

            x_std = F.relu(self.lin_out_std1(g_squeeze))
            x_1_std = F.relu(self.lin_out_std2(x_std))
                           
            # g_2 = F.relu(self.lin_out_std(g_squeeze_flat))

            # x_1_std = g_2.reshape(BATCH_N, MAX_N, -1)

            return {'mu' : x_1, 'std' : x_1_std}
        else:
            return x_1

