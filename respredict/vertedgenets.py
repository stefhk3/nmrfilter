
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from nets import * 

class VLayers(nn.Module):
    """
    BASELINE
    """
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 norm=None,
                 force_use_bias = False, 
                 noise=1e-5, agg_func=None,
                 layer_class = 'GraphMatLayerFast', 
                 layer_config = {}):
        super(VLayers, self).__init__()
        
        self.gl = nn.ModuleList()
        self.resnet = resnet

        LayerClass = eval(layer_class)
        for li in range(len(output_features_n)):
            if li == 0:
                gl = LayerClass(input_feature_n, output_features_n[0],
                                noise=noise, agg_func=agg_func, GS=GS, 
                                use_bias=not norm or force_use_bias, 
                                **layer_config)
            else:
                gl = LayerClass(output_features_n[li-1], 
                                output_features_n[li], 
                                noise=noise, agg_func=agg_func, GS=GS, 
                                use_bias=not norm or force_use_bias, 
                                **layer_config)
            
            self.gl.append(gl)

        self.norm = norm
        if self.norm is not None:
            if self.norm == 'batch':
                Nlayer = MaskedBatchNorm1d
            elif self.norm == 'layer':
                Nlayer = MaskedLayerNorm1d
            self.bn = nn.ModuleList([Nlayer(f) for f in output_features_n])
            
        
    def forward(self, G, x, edge_edge, edge_vert, edge_feat, input_mask=None):
        # print("G.shape=", G.shape,
        #       "x.shape=", x.shape,
        #       "edge_edge.shape=", edge_edge.shape,
        #       "edge_vert.shape=", edge_vert.shape,
        #       "edge_feat.shape=", edge_feat.shape)
        for gi, gl in enumerate(self.gl):
            x2 = gl(G, x)
            if self.norm:
                x2 = self.bn[gi](x2.reshape(-1, x2.shape[-1]), 
                                 input_mask.reshape(-1)).reshape(x2.shape)

            if self.resnet:
                if x.shape == x2.shape:
                    x3 = x2 + x
                else:
                    x3 = x2
            else:
                x3 = x2
            x = x3
        

        return x

    
class VELayers(nn.Module):
    def __init__(self, input_feature_n, 
                 output_features_n,
                 resnet=False, GS=1, 
                 norm=None,
                 force_use_bias = False, 
                 noise=1e-5, agg_func=None,
                 layer_class = 'GraphMatLayerFast', 
                 layer_config = {}):
        super(VELayers, self).__init__()
        
        self.gl_ve = nn.ModuleList()
        self.gl_ev = nn.ModuleList()
        
        self.resnet = resnet

        self.feature_n = output_features_n[-1]
        feature_n = self.feature_n
        
        layer_n = len(output_features_n)

        LayerClass = eval(layer_class)
        for li in range(layer_n):
            gl = LayerClass(feature_n, feature_n, 
                            noise=noise, agg_func=agg_func, GS=1, 
                            use_bias=not norm or force_use_bias, 
                            **layer_config)
            
            self.gl_ve.append(gl)

            gl = LayerClass(feature_n + 4, feature_n, 
                            noise=noise, agg_func=agg_func, GS=1, 
                            use_bias=not norm or force_use_bias, 
                            **layer_config)
            
            self.gl_ev.append(gl)
            
            

        self.norm = norm
        if self.norm is not None:
            if self.norm == 'batch':
                Nlayer = MaskedBatchNorm1d
            elif self.norm == 'layer':
                Nlayer = MaskedLayerNorm1d
            self.bn = nn.ModuleList([Nlayer(f) for f in output_features_n])
            
        
    def forward(self, G, vert_feat_in, edge_edge, edge_vert, edge_feat_in, input_mask=None):
        # print("G.shape=", G.shape,
        #       "vert_feat_in.shape=", vert_feat_in.shape,
        #       "edge_edge.shape=", edge_edge.shape,
        #       "edge_vert.shape=", edge_vert.shape,
        #       "edge_feat.shape=", edge_feat.shape)

        vert_feat = F.pad(vert_feat_in,
                          (0, self.feature_n - vert_feat_in.shape[-1]) ,
                          "constant", 0)


        for gi, (gl_ve, gl_ev) in enumerate(zip(self.gl_ve, self.gl_ev)):

            edge_feat = gl_ve(edge_vert, vert_feat)

            vert_feat_next = gl_ev(edge_vert.permute(0, 1, 3, 2),
                                   torch.cat([edge_feat, edge_feat_in], -1))
            vert_feat_next = self.bn[gi](vert_feat_next.reshape(-1, vert_feat_next.shape[-1]), 
                                         input_mask.reshape(-1)).reshape(vert_feat_next.shape)

            if self.resnet:
                
                vert_feat = vert_feat_next + vert_feat
            else:
                vert_feat = vert_feat_next
        # for gi, gl in enumerate(self.gl):
        #     x2 = gl(G, x)
        #     if self.norm:
        #         x2 = self.bn[gi](x2.reshape(-1, x2.shape[-1]), 
        #                          input_mask.reshape(-1)).reshape(x2.shape)

        #     If self.resnet:
        #         if x.shape == x2.shape:
        #             x3 = x2 + x
        #         else:
        #             x3 = x2
        #     else:
        #         x3 = x2
        #     x = x3
        

        return vert_feat

    
class VEExtraResLayers(nn.Module):
    def __init__(self, input_feature_n, 
                 output_features_n,
                 resnet=False, GS=1, 
                 norm=None,
                 force_use_bias = False, 
                 noise=1e-5, agg_func=None,
                 layer_class = 'GraphMatLayerFast', 
                 layer_config = {}):
        super(VEExtraResLayers, self).__init__()
        
        self.gl_ve = nn.ModuleList()
        self.gl_ev = nn.ModuleList()
        
        self.resnet = resnet

        self.feature_n = output_features_n[-1]
        feature_n = self.feature_n
        
        layer_n = len(output_features_n)

        LayerClass = eval(layer_class)
        for li in range(layer_n):
            gl = LayerClass(feature_n, feature_n, 
                            noise=noise, agg_func=agg_func, GS=1, 
                            use_bias=not norm or force_use_bias, 
                            **layer_config)
            
            self.gl_ve.append(gl)

            gl = LayerClass(feature_n + 4, feature_n, 
                            noise=noise, agg_func=agg_func, GS=1, 
                            use_bias=not norm or force_use_bias, 
                            **layer_config)
            
            self.gl_ev.append(gl)
            
            

        self.norm = norm
        if self.norm is not None:
            if self.norm == 'batch':
                Nlayer = MaskedBatchNorm1d
            elif self.norm == 'layer':
                Nlayer = MaskedLayerNorm1d
            self.bn = nn.ModuleList([Nlayer(f) for f in output_features_n])
            
        
    def forward(self, G, vert_feat_in, edge_edge, edge_vert, edge_feat_in, input_mask=None):
        # print("G.shape=", G.shape,
        #       "vert_feat_in.shape=", vert_feat_in.shape,
        #       "edge_edge.shape=", edge_edge.shape,
        #       "edge_vert.shape=", edge_vert.shape,
        #       "edge_feat.shape=", edge_feat.shape)

        vert_feat = F.pad(vert_feat_in,
                          (0, self.feature_n - vert_feat_in.shape[-1]) ,
                          "constant", 0)


        for gi, (gl_ve, gl_ev) in enumerate(zip(self.gl_ve, self.gl_ev)):

            edge_feat_next =  gl_ve(edge_vert, vert_feat)

            # edge_feat_next = self.edge_bn[gi](edge_feat_next.reshape(-1, edge_feat_next.shape[-1]), 
            #                              input_mask.reshape(-1)).reshape(vert_feat_next.shape)
            
            if self.resnet and gi > 0:
                edge_feat = edge_feat + edge_feat_next
            else:
                edge_feat = edge_feat_next

            vert_feat_next = gl_ev(edge_vert.permute(0, 1, 3, 2),
                                   torch.cat([edge_feat, edge_feat_in], -1))
            vert_feat_next = self.vert_bn[gi](vert_feat_next.reshape(-1, vert_feat_next.shape[-1]), 
                                         input_mask.reshape(-1)).reshape(vert_feat_next.shape)

            if self.resnet:
                
                vert_feat = vert_feat_next + vert_feat
            else:
                vert_feat = vert_feat_next
        # for gi, gl in enumerate(self.gl):
        #     x2 = gl(G, x)
        #     if self.norm:
        #         x2 = self.bn[gi](x2.reshape(-1, x2.shape[-1]), 
        #                          input_mask.reshape(-1)).reshape(x2.shape)

        #     If self.resnet:
        #         if x.shape == x2.shape:
        #             x3 = x2 + x
        #         else:
        #             x3 = x2
        #     else:
        #         x3 = x2
        #     x = x3
        

        return vert_feat

    
class VEBootstrap(nn.Module):
    def __init__(self, g_feature_n, g_feature_out_n=None, 
                 int_d = None, layer_n = None, 
                 mixture_n = 5, 
                 resnet=True, 
                 gml_class = 'GraphMatLayers',
                 gml_config = {}, 
                 init_noise=1e-5,
                 init_bias = 0.0, agg_func=None, GS=1, OUT_DIM=1, 
                 input_norm='batch', out_std= False, 
                 resnet_out = False, resnet_blocks = (3, ), 
                 resnet_d = 128, 
                 inner_norm=None, 
                 out_std_exp = False, 
                 force_lin_init=False, 
                 use_random_subsets=True,
                 out_scale = 1.0):
        
        """

        
        """
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n

        super( VEBootstrap, self).__init__()
        self.gml = eval(gml_class)(g_feature_n, g_feature_out_n, 
                                   resnet=resnet, noise=init_noise,
                                   agg_func=parse_agg_func(agg_func), 
                                   norm = inner_norm, 
                                   GS=GS,
                                   **gml_config)

        if input_norm == 'batch':
            self.input_norm = MaskedBatchNorm1d(g_feature_n)
        elif input_norm == 'layer':
            self.input_norm = MaskedLayerNorm1d(g_feature_n)
        else:
            self.input_norm = None

        self.resnet_out = resnet_out 
        if not resnet_out:
            self.mix_out = nn.ModuleList([nn.Linear(g_feature_out_n[-1], OUT_DIM) for _ in range(mixture_n)])
        else:
            self.mix_out = nn.ModuleList([ResNetRegressionMaskedBN(g_feature_out_n[-1], 
                                                           block_sizes = resnet_blocks, 
                                                           INT_D = resnet_d, 
                                                           FINAL_D=resnet_d, 
                                                           OUT_DIM=OUT_DIM) for _ in range(mixture_n)])

        self.out_std = out_std
        self.out_std_exp = False
        self.out_scale = out_scale
        
        self.use_random_subsets = use_random_subsets

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        if init_bias > 0:
                            nn.init.normal_(m.bias, 0, init_bias)
                        else:
                            nn.init.constant_(m.bias, 0)

    def forward(self, adj, vect_feat, input_mask, input_idx, adj_oh,
                return_g_features = False, also_return_g_features = False,
                edge_edge = None, edge_vert = None, edge_feat = None, 
                **kwargs):

        G = adj
        
        BATCH_N, MAX_N, F_N = vect_feat.shape

        if self.input_norm is not None:
            vect_feat = apply_masked_1d_norm(self.input_norm, 
                                             vect_feat, 
                                             input_mask)
        
        G_features = self.gml(G, vect_feat, edge_edge,
                              edge_vert, edge_feat, input_mask)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])
        
        if self.resnet_out:
            x_1 = [m(g_squeeze_flat, input_mask.reshape(-1)).reshape(BATCH_N, MAX_N, -1) for m in self.mix_out]
        else:
            x_1 = [m(g_squeeze) for m in self.mix_out]

        x_1 = torch.stack(x_1) * self.out_scale

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
        else:
            x_1_sub = x_1.mean(dim=0)
        # #print(x_1.shape)
        # idx = torch.randint(high=x_1.shape[0], 
        #                     size=(BATCH_N, )).to(G.device)
        # #print("idx=", idx)
        # x_1_sub = torch.stack([x_1[v, v_i] for v_i, v in enumerate(idx)])
        std = torch.sqrt(torch.var(x_1, dim=0) + 1e-5)

        #print("numpy_std=", np.std(x_1.detach().cpu().numpy()))

        #kjlkjalds
        x_1 = x_1_sub

        ret = {'mu' : x_1, 'std' : std}
        if also_return_g_features:
            ret['g_features'] = g_squeeze
        return ret

