from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import pickle
import copy 

def goodmax(x, dim):
    return torch.max(x, dim=dim)[0]


def create_mp(MAX_N, F, STEP_N, args):
    args = copy.deepcopy(args)
    class_name = args['name']
    del args['name']
    return eval(class_name)(MAX_N = MAX_N, F=F, 
                            STEP_N=STEP_N, **args)
    

class Vpack:
    def __init__(self, BATCH_N, MAX_N, F, mask):
        """
        
        """
        self.BATCH_N = BATCH_N
        self.F = F
        self.MAX_N = MAX_N 
        self.mask = mask
        
    def zero(self, V):
        mask = self.mask.reshape(-1).unsqueeze(-1)
        return (V.reshape(-1, self.F) * mask).reshape(V.shape)
    
    def pack(self, V):
        V_flat = V.reshape(-1, self.F)
        mask = (self.mask>0).reshape(-1)
        return V_flat[mask]
    
    def unpack(self, V):
        output = torch.zeros((self.BATCH_N *self.MAX_N, V.shape[-1]), device=V.device)
        mask = (self.mask>0).reshape(-1)
        output[mask] = V
        return output.reshape(self.BATCH_N, self.MAX_N, V.shape[-1])
    
    
class Epack:
    def __init__(self, BATCH_N, MAX_N, F, mask):
        """
        
        """
        self.BATCH_N = BATCH_N
        self.F = F
        self.MAX_N = MAX_N 
        self.mask = mask
        
    def zero(self, E):
        mask = self.mask.reshape(-1).unsqueeze(-1)
        return (E.reshape(-1, self.F) * mask).reshape(E.shape)
    
    def pack(self, E):
        E_flat = E.reshape(-1, self.F)
        mask = (self.mask>0).reshape(-1)
        return E_flat[mask]
    
    def unpack(self, E):
        output = torch.zeros((self.BATCH_N * self.MAX_N * self.MAX_N, E.shape[-1]), 
                             device=E.device)
        mask = (self.mask>0).reshape(-1)
        output[mask] = E
        return output.reshape(self.BATCH_N, self.MAX_N, self.MAX_N, E.shape[-1])
    

class MPMLPOutNet(nn.Module):
    def __init__(self, vert_f_in, edge_f_in, MAX_N, 
                 layer_n, internal_d_vert,
                 internal_d_edge, init_noise=0.01, 
                 force_lin_init=False, dim_out=4, 
                 final_d_out = 1024, 
                 final_layer_n = 1, 
                 final_norm = 'batch',
                 force_bias_zero=True, 
                 layer_use_bias = True, 
                 edge_mat_norm = False, 
                 force_edge_zero = False, 
                 mpconfig = {},
                 logsoftmax_out = False, 
                 chan_out = 1, 
                 vert_mask_use = True, 
                 edge_mask_use = True,
                 pos_out = False, 
                 vert_bn_use = True, 
                 edge_bn_use = True, 
                 mask_val_edges=False, 
                 combine_graph_in = False, 
                 log_invalid_offset = -1e4, 
                 e_bn_pre_mlp = False
                 
):
        """
        MPNet but with a MP output 

        """
        
        
        super(MPMLPOutNet, self).__init__()
        
        self.MAX_N = MAX_N
        self.vert_f_in = vert_f_in
        self.edge_f_in = edge_f_in
        
        self.dim_out = dim_out
        self.internal_d_vert = internal_d_vert
        self.layer_n = layer_n
        self.edge_mat_norm = edge_mat_norm
        self.force_edge_zero = force_edge_zero

        self.input_v_bn = MaskedBatchNorm1d(vert_f_in)
        self.input_e_bn = MaskedBatchNorm1d(edge_f_in)

        self.mp = create_mp(MAX_N, internal_d_vert, 
                            layer_n, mpconfig)

        self.combine_graph_in = combine_graph_in
        extra_mlp_dim = 0
        if self.combine_graph_in:
            extra_mlp_dim = 4
        self.final_mlp_out = graphnet2.MLPModel(final_d_out, 
                                                final_layer_n, 
                                                input_d = internal_d_vert  + extra_mlp_dim)

        self.e_bn_pre_mlp = e_bn_pre_mlp
        if e_bn_pre_mlp:
            self.pre_mlp_bn = MaskedBatchNorm1d(internal_d_vert  + extra_mlp_dim)
        self.final_out = nn.Linear(final_d_out, 
                                   dim_out)
        # self.per_e_l = nn.Linear(internal_d_vert, final_d_out)

        # self.per_e_l_1 = nn.Linear(final_d_out, final_d_out)
        self.chan_out = chan_out
        # self.per_e_out = nn.ModuleList([nn.Linear(final_d_out, dim_out) for _ in range(self.chan_out)])
        
        self.init_noise = init_noise

        if force_lin_init:
            self.force_init(init_noise, force_bias_zero)


        self.triu_idx = torch.Tensor(triu_indices_flat(MAX_N, k=1)).long()


        #self.softmax_out = softmax_out
        self.logsoftmax_out = logsoftmax_out 
        self.pos_out = pos_out

        self.vert_mask_use = vert_mask_use
        self.edge_mask_use = edge_mask_use

        self.vert_bn_use = vert_bn_use
        self.edge_bn_use = edge_bn_use
        self.mask_val_edges = mask_val_edges
        self.log_invalid_offset = log_invalid_offset


    def force_init(self, init_noise=None, force_bias_zero=True):
        if init_noise is None:
            init_noise = self.init_noise
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_noise < 1e-12:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    
                    nn.init.normal_(m.weight, 0, init_noise)
                if force_bias_zero:
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


    def forward(self, v_in, e_in, graph_conn_in, out_mask, 
                vert_mask, possible_edge_prior, possible_val, *args):
        """
        output is:
        [BATCH_N, FLATTEN_LENGHED_N, LABEL_LEVELS, M]
        """

        BATCH_N = v_in.shape[0]
        MAX_N = v_in.shape[1]

        v_mask = vert_mask.unsqueeze(-1)
        e_mask = (vert_mask.unsqueeze(-1) * vert_mask.unsqueeze(-2)).unsqueeze(-1)

        def v_osum(v):
            return v.unsqueeze(1) + v.unsqueeze(2)
            
        def last_bn(layer, x, mask):
            init_shape = x.shape

            x_flat = x.reshape(-1, init_shape[-1])
            mask_flat = mask.reshape(-1)
            x_bn = layer(x_flat, mask_flat)
            return x_bn.reshape(init_shape)

        def combine_vv(li, v1, v2):
            if self.bilinear_vv:
                return self.lin_vv_layers[li](v1, v2)
            elif self.gru_vv:
                if self.gru_ab:
                    a,b = v1, v2 
                else:
                    a,b = v2, v1
                return self.lin_vv_layers[li](a.reshape(-1, a.shape[-1]), 
                                              b.reshape(-1, b.shape[-1]))\
                           .reshape(a.shape)
            else:
                return self.lin_vv_layers[li](torch.cat([v1, v2], dim=-1))

        f1 = torch.relu
        f2 = torch.relu


        ### DEBUG FORCE TO ZERO
        if self.force_edge_zero:
            e_in[:] = 0  

        if self.edge_mat_norm:
            e_in = batch_mat_chan_norm(e_in)

        def resnet_mod(i, k):
            if k > 0:
                if (i % k) == k-1:
                    return True
            return False

        v_in_bn = last_bn(self.input_v_bn, v_in, v_mask)
        e_in_bn = last_bn(self.input_e_bn, e_in, e_mask)

        v = F.pad(v_in_bn, (0, self.internal_d_vert - v_in_bn.shape[-1]) , "constant", 0)

        if self.vert_mask_use:
            v = v * v_mask

        e = F.pad(e_in_bn, 
                  (0, self.internal_d_vert - e_in_bn.shape[-1]) ,
                  "constant", 0)
        if self.edge_mask_use :
            e = e * e_mask

        e_new, v = self.mp(e, v, e_mask, v_mask)

        if self.combine_graph_in:
            e_new = torch.cat([e_new, graph_conn_in], -1)
        if self.e_bn_pre_mlp :
            e_new = last_bn(self.pre_mlp_bn, e_new, e_mask)

        e_est = self.final_out(self.final_mlp_out(e_new))

        e_est = e_est.unsqueeze(-1)
        # #print("v_new.shape=", v_new.shape)
        # e_est_int = torch.relu(self.per_e_l(e_new))
        # e_est_int = torch.relu(self.per_e_l_1(e_est_int))

        # e_est = torch.stack([e(e_est_int) for e in self.per_e_out], -1)
        
        assert e_est.shape[-1] == self.chan_out

        ##multi_e_out = multi_e_out.squeeze(-2)
        if self.mask_val_edges:
            e_est = e_est + (1.0 - possible_val.unsqueeze(-1))*self.log_invalid_offset
        
        a_flat = e_est.reshape(BATCH_N, -1, self.dim_out, self.chan_out)
        #print("a_flat.shape=", a_flat.shape)
        a_triu_flat = a_flat[:, self.triu_idx, :, :]
        
        if self.logsoftmax_out:
            # SOFTMAX_OFFSET = -1e2
            # if out_mask is not None:
            #     out_mask_offset = SOFTMAX_OFFSET * (1-out_mask.unsqueeze(-1).unsqueeze(-1))
            #     a_triu_flat += out_mask_offset
            a_triu_flatter = a_triu_flat.reshape(BATCH_N, -1, 1)
            if self.logsoftmax_out:
                a_nonlin = F.log_softmax(a_triu_flatter, dim=1)
            elif self.softmax_out:
                a_nonlin = F.softmax(a_triu_flatter, dim=1)
            else:
                raise ValueError()
                
            a_nonlin = a_nonlin.reshape(BATCH_N, -1, self.dim_out, 1)
        else:
        
            a_nonlin = a_triu_flat
        
        if self.pos_out:
            a_nonlin = F.relu(a_nonlin)

        return a_nonlin


class EVMPVaryVertCombine(nn.Module):
    """
    Experiment with different ways of combining the vertices

    """
    def __init__(self, MAX_N, F, STEP_N, celltype='CustomGRU', 
                 vert_combine = 'prod', return_intermediate_e = False):
        super(EVMPVaryVertCombine, self).__init__()
        
        self.MAX_N = MAX_N
        self.F = F
        print(f"EVMPVaryVerTcombine, MAX_N={MAX_N}, F={F}, STEP_N={STEP_N}")
        
        if celltype == 'CustomGRU':
            ct  = CustomGRU
        elif celltype == 'SimpleCell':
            ct = SimpleCell
        elif celltype == 'DebugCell':
            ct = DebugCell
        elif celltype == 'ExtraProductCell':
            ct = ExtraProductCell
        elif celltype == 'SimpleCell':
            ct = SimpleCell
        elif celltype == 'TrivialCell':
            ct = TrivialCell
        else:
            raise ValueError(f"unknown type {celltype}")
        self.e_cell = ct(F, F)
        self.v_cell = ct(F, F)

        self.STEP_N = STEP_N
        
        self.vert_combine = vert_combine
        self.return_intermediate_e = return_intermediate_e

    def forward(self, e, v, e_mask, v_mask):
        BATCH_N = e.shape[0]
        
        
        vp = Vpack(BATCH_N, self.MAX_N, self.F, v_mask)
        ep = Epack(BATCH_N, self.MAX_N, self.F, e_mask)


        v_in_f = vp.pack(v)
        e_in_f = ep.pack(e)
        
        v_h = v_in_f
        e_h = e_in_f
        
        e_v = torch.zeros_like(v_h)
        intermediate_e_h_up = []
        
        for i in range(self.STEP_N):
            v_h = self.v_cell(e_v, v_h)
            v_h_up = vp.unpack(v_h) 
            if self.vert_combine == 'prod':
                v_e = v_h_up.unsqueeze(1)  * v_h_up.unsqueeze(2)
            elif self.vert_combine == 'sum':
                v_e = v_h_up.unsqueeze(1)  + v_h_up.unsqueeze(2)

            elif self.vert_combine == 'relu_sum':
                v_e = F.relu(v_h_up.unsqueeze(1))  + F.relu(v_h_up.unsqueeze(2))
            elif self.vert_combine == 'relu_prod':
                v_e = F.relu(v_h_up.unsqueeze(1)) * F.relu(v_h_up.unsqueeze(2))
            elif self.vert_combine == 'softmax_prod':
                vs = torch.softmax(v_h_up, dim=-1)
                v_e = vs.unsqueeze(1) * vs.unsqueeze(2)
            elif self.vert_combine == 'max':
                v_e = torch.max(v_h_up.unsqueeze(1), v_h_up.unsqueeze(2))

            else:
                raise ValueError(f"Unknown vert_combine {self.vert_combine}")

            v_e_p = ep.pack(v_e)
            e_h = self.e_cell(v_e_p, e_h)
            e_h_up = ep.unpack(e_h)
            intermediate_e_h_up.append(e_h_up)
            e_v_up = goodmax(e_h_up, dim=1)
                

            e_v = vp.pack(e_v_up)
        if self.return_intermediate_e:
            return e_h_up, v_h_up, intermediate_e_h_up
        else:
            return ep.unpack(e_h), v_h_up




class SimpleCell(nn.Module):
    def __init__(self, D, _):
        super(SimpleCell, self).__init__()

        self.Lrx = nn.Linear(D, D)
        self.Lrh = nn.Linear(D, D)
    
    def forward(self, x, h):
        r = F.sigmoid(self.Lrx(x) + self.Lrh(h))
        return r
    
class CustomGRU(nn.Module):
    def __init__(self, D, _):
        super(CustomGRU, self).__init__()

        self.Lrx = nn.Linear(D, D)
        self.Lrh = nn.Linear(D, D)
        
        self.Lzx = nn.Linear(D, D)
        self.Lzh = nn.Linear(D, D)
        
        self.Lnx = nn.Linear(D, D)
        self.Lnh = nn.Linear(D, D)
    
    def forward(self, x, h):
        r = torch.sigmoid(self.Lrx(x) + self.Lrh(h))
        z = torch.sigmoid(self.Lzx(x) + self.Lzh(h))
        n = torch.tanh(self.Lnx(x) + r*self.Lnh(h))
        
        return (1-z) * n + z * h

class EVMPMultiLayer(nn.Module):
    """
    Experiment with different ways of combining the vertices

    """
    def __init__(self, MAX_N, F, STEP_N, celltype='CustomGRU', 
                 vert_combine = 'prod', return_intermediate_e = False):
        super(EVMPMultiLayer, self).__init__()
        
        self.MAX_N = MAX_N
        self.F = F
        print(f"EVMPMultiLayercombine, MAX_N={MAX_N}, F={F}, STEP_N={STEP_N}")
        
        if celltype == 'CustomGRU':
            ct  = CustomGRU
        elif celltype == 'SimpleCell':
            ct = SimpleCell
        elif celltype == 'DebugCell':
            ct = DebugCell
        elif celltype == 'ExtraProductCell':
            ct = ExtraProductCell
        elif celltype == 'SimpleCell':
            ct = SimpleCell
        elif celltype == 'TrivialCell':
            ct = TrivialCell
        else:
            raise ValueError(f"unknown type {celltype}")
        self.e_cells = nn.ModuleList([ct(F, F) for _ in range(STEP_N)])
        self.v_cells = nn.ModuleList([ct(F, F) for _ in range(STEP_N)])

        self.STEP_N = STEP_N
        
        self.vert_combine = vert_combine
        self.return_intermediate_e = return_intermediate_e

    def forward(self, e, v, e_mask, v_mask):
        BATCH_N = e.shape[0]
        
        
        vp = Vpack(BATCH_N, self.MAX_N, self.F, v_mask)
        ep = Epack(BATCH_N, self.MAX_N, self.F, e_mask)


        v_in_f = vp.pack(v)
        e_in_f = ep.pack(e)
        
        v_h = v_in_f
        e_h = e_in_f
        
        e_v = torch.zeros_like(v_h)
        intermediate_e_h_up = []
        
        for i in range(self.STEP_N):
            v_h = self.v_cells[i](e_v, v_h)
            v_h_up = vp.unpack(v_h) 
            if self.vert_combine == 'prod':
                v_e = v_h_up.unsqueeze(1)  * v_h_up.unsqueeze(2)
            elif self.vert_combine == 'sum':
                v_e = v_h_up.unsqueeze(1)  + v_h_up.unsqueeze(2)

            elif self.vert_combine == 'relu_sum':
                v_e = F.relu(v_h_up.unsqueeze(1))  + F.relu(v_h_up.unsqueeze(2))
            elif self.vert_combine == 'relu_prod':
                v_e = F.relu(v_h_up.unsqueeze(1)) * F.relu(v_h_up.unsqueeze(2))
            elif self.vert_combine == 'softmax_prod':
                vs = torch.softmax(v_h_up, dim=-1)
                v_e = vs.unsqueeze(1) * vs.unsqueeze(2)
            elif self.vert_combine == 'max':
                v_e = torch.max(v_h_up.unsqueeze(1), v_h_up.unsqueeze(2))

            else:
                raise ValueError(f"Unknown vert_combine {self.vert_combine}")

            v_e_p = ep.pack(v_e)
            e_h = self.e_cells[i](v_e_p, e_h)
            e_h_up = ep.unpack(e_h)
            intermediate_e_h_up.append(e_h_up)
            e_v_up = goodmax(e_h_up, dim=1)
                

            e_v = vp.pack(e_v_up)
        if self.return_intermediate_e:
            return e_h_up, v_h_up, intermediate_e_h_up
        else:
            return ep.unpack(e_h), v_h_up


class ExtraProductCell(nn.Module):
    """
    GRU cell but where we also mask by x, 
    creating a second 'r' (called r2)
    """
    def __init__(self, D, _):
        super(ExtraProductCell, self).__init__()

        self.Lrx = nn.Linear(D, D)
        self.Lrh = nn.Linear(D, D)
        
        self.Lr2x = nn.Linear(D, D)
        self.Lr2h = nn.Linear(D, D)
        
        self.Lzx = nn.Linear(D, D)
        self.Lzh = nn.Linear(D, D)
        
        self.Lnx = nn.Linear(D, D)
        self.Lnh = nn.Linear(D, D)
        
    
    def forward(self, x, h):
        r = torch.sigmoid(self.Lrx(x) + self.Lrh(h))
        r2 = torch.sigmoid(self.Lr2x(x) + self.Lr2h(h))
        z = torch.sigmoid(self.Lzx(x) + self.Lzh(h))
        n = torch.tanh(r2*self.Lnx(x) + r*self.Lnh(h))
        
        return (1-z) * n + z * h
        
class GraphMPLayers(nn.Module):
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 norm=None,
                 force_use_bias = False, 
                 noise=1e-5, agg_func=None,
                 layer_class = 'GraphMatLayerFast', 
                 layer_config = {}):
        super(GraphMPLayers, self).__init__()


        ### where does MAX_N come from ?
        self.internal_d = output_features_n[-1]
        MAX_N = layer_config['MAX_N']
        step_n = layer_config['step_n']
        del layer_config['MAX_N']
        del layer_config['step_n']
        print('layer config=', layer_config)
        self.mp = create_mp(MAX_N, 
                            self.internal_d, 
                            step_n, 
                            layer_config)
        
        
        
    def forward(self, G, x, input_mask=None):

        ## pad G and X
        e_in = G.permute(0, 2, 3, 1)

        v_mask = input_mask.unsqueeze(-1)
        e_mask = input_mask.unsqueeze(1) * input_mask.unsqueeze(2)

        #print(f"v_mask.shape={v_mask.shape}, e_mask.shape={e_mask.shape}")
        v_in = x

        v = F.pad(v_in, (0, self.internal_d - v_in.shape[-1]) ,
                  "constant", 0)

        v = v * v_mask

        e = F.pad(e_in, 
                  (0, self.internal_d - e_in.shape[-1]) ,
                  "constant", 0)

        
        e_new, v = self.mp(e, v, e_mask, v_mask)

        return v




