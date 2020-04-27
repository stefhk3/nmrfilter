
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nets

class GraphEdgeVertLayer(nn.Module):
    def __init__(self,vert_f_in, vert_f_out, 
                 edge_f_in, edge_f_out, 
                 #e_agg = torch.sum, 
                 e_agg = nets.goodmax, 
                 out_func = F.relu):
        """
        Note that to do the per-edge-combine-vertex layer we
        apply a per-vertex linear layer first and sum the result
        to the edge layer
        
        FIXME throw in some batch norms and some resnets because
            that appears to be a thing people do 
        """
        
        
        super(GraphEdgeVertLayer, self).__init__()
        
        self.edge_f_in = edge_f_in
        self.edge_f_out = edge_f_out
        self.vert_f_in = vert_f_in
        self.vert_f_out = vert_f_out
        
        self.e_vert_layer = nn.Linear(self.vert_f_in, self.edge_f_out)
        self.e_layer = nn.Linear(self.edge_f_in, self.edge_f_out)
        
        self.v_layer = nn.Linear(self.vert_f_in + self.edge_f_out, 
                                self.vert_f_out)
        
        self.e_agg = e_agg
    
    def forward(self, v_in, e_in):
        BATCH_N = v_in.shape[0]
        assert v_in.shape[0] == e_in.shape[0]
        
        ### per-edge operations
        e_v = self.e_vert_layer(v_in)
        outer_v_sum = e_v.unsqueeze(1) + e_v.unsqueeze(2)
        e = self.e_layer(e_in)
        e_out = F.relu(e + outer_v_sum)
        
        ### per-vertex operations
        per_v_e = self.e_agg(e_out, dim=1)
        vert_e_combined = torch.cat([per_v_e, v_in], dim=2)
        v = self.v_layer(vert_e_combined)
        v_out = F.relu(v)
        
        return v_out, e_out
        
    



class GraphEdgeVertPred(nn.Module):
    def __init__(self, f_in_vert, f_in_edge, 
                 MAX_N, layer_n, internal_d_vert, internal_d_edge, resnet=False, 
                 INIT_NOISE = 0.01, OUT_DIM=1):
        """
        Note that to do the per-edge-combine-vertex layer we
        apply a per-vertex linear layer first and sum the result
        to the edge layer
        
        FIXME throw in some batch norms and some resnets because
            that appears to be a thing people do 
        """
        
        
        super(GraphEdgeVertPred, self).__init__()
        
        self.MAX_N = MAX_N
        self.f_in_vert = f_in_vert
        self.f_in_edge = f_in_edge
        self.g1 = GraphEdgeVertLayer(f_in_vert, internal_d_vert, 
                                     f_in_edge, internal_d_edge)
        self.g_inner = nn.ModuleList([GraphEdgeVertLayer(internal_d_vert, 
                                                         internal_d_vert, 
                                                         internal_d_edge, 
                                                         internal_d_edge) for _ in range(layer_n)])
        #self.g3 = GraphEdgeVertLayer(internal_d_vert, 1, internal_d_edge, 1)
        
        self.as_resnet = resnet
        
        #self.bn_v_in = nn.BatchNorm1d(MAX_N * f_in_vert)
        #self.bn_v_inner = nn.ModuleList([nn.BatchNorm1d(MAX_N, internal_d_vert) for _ in range(layer_n)])

        self.outLin = nn.Linear(internal_d_vert, OUT_DIM)

        self.init_noise = INIT_NOISE
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.bias, -self.init_noise, self.init_noise)
                nn.init.uniform_(m.weight, -self.init_noise, self.init_noise)

    def forward(self, args):
        e_in, v, m = args

        e = torch.transpose(e_in, 1,3)
        e = torch.cat([e, m], dim=-1)
        
        #v = self.bn_v_in(v.view(-1, self.MAX_N * self.f_in_vert))
        #v = v.view(-1, self.MAX_N, self.f_in_vert)
        #print("v.shape=", v.shape, "e.shape=", e.shape)
        v, e = self.g1(v, e)
        for i in range(len(self.g_inner)):
        
            v_next, e_next = self.g_inner[i](v, e)
            if self.as_resnet:
                v = v_next + v
                e = e_next + e
            else:
                v = v_next
                e = e_next
            #v = self.bn_v_inner[i](v)
        #v, e = self.g3(v, e)
        #print("v.shape=", v.shape, "e.shape=", e.shape)
        return self.outLin(v)


class GraphEdgeVertPredHighway(nn.Module):
    def __init__(self, f_in_vert, f_in_edge, 
                 MAX_N, layer_n, internal_d_vert, internal_d_edge, resnet=False, 
                 INIT_NOISE = 0.01):
        """
        Note that to do the per-edge-combine-vertex layer we
        apply a per-vertex linear layer first and sum the result
        to the edge layer
        
        FIXME throw in some batch norms and some resnets because
            that appears to be a thing people do 
        """
        
        
        super(GraphEdgeVertPredHighway, self).__init__()
        
        self.MAX_N = MAX_N
        self.f_in_vert = f_in_vert
        self.f_in_edge = f_in_edge
        self.g1 = GraphEdgeVertLayer(f_in_vert, internal_d_vert, 
                                     f_in_edge, internal_d_edge)
        self.g_inner = nn.ModuleList([GraphEdgeVertLayer(internal_d_vert, 
                                                         internal_d_vert, 
                                                         internal_d_edge, 
                                                         internal_d_edge) for _ in range(layer_n)])
        #self.g3 = GraphEdgeVertLayer(internal_d_vert, 1, internal_d_edge, 1)
        
        self.as_resnet = resnet
        
        #self.bn_v_in = nn.BatchNorm1d(MAX_N * f_in_vert)
        #self.bn_v_inner = nn.ModuleList([nn.BatchNorm1d(MAX_N, internal_d_vert) for _ in range(layer_n)])

        self.outLin = nn.Linear(internal_d_vert, 1)

        self.init_noise = INIT_NOISE
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.bias, -self.init_noise, self.init_noise)
                nn.init.uniform_(m.weight, -self.init_noise, self.init_noise)

    def forward(self, args):
        e, v = args
        e = torch.transpose(e, 1,3)
        
        #v = self.bn_v_in(v.view(-1, self.MAX_N * self.f_in_vert))
        #v = v.view(-1, self.MAX_N, self.f_in_vert)
        #print("v.shape=", v.shape, "e.shape=", e.shape)
        highway = []
        v, e = self.g1(v, e)
        highway.append(v)
        for i in range(len(self.g_inner)):
        
            v_next, e_next = self.g_inner[i](v, e)
            if self.as_resnet:
                v = v_next + v
                e = e_next + e
            else:
                v = v_next
                e = e_next
            highway.append(v)
            #v = self.bn_v_inner[i](v)
        #v, e = self.g3(v, e)
        #print("v.shape=", v.shape, "e.shape=", e.shape)
        v_highway = F.relu(sum(highway))
        return self.outLin(v_highway)


class GraphEdgeVertPredHighway2(nn.Module):
    def __init__(self, f_in_vert, f_in_edge, 
                 MAX_N, layer_n, internal_d_vert, internal_d_edge, resnet=False, 
                 INIT_NOISE = 0.01):
        """
        Note that to do the per-edge-combine-vertex layer we
        apply a per-vertex linear layer first and sum the result
        to the edge layer
        
        FIXME throw in some batch norms and some resnets because
            that appears to be a thing people do 
        """
        
        
        super(GraphEdgeVertPredHighway2, self).__init__()
        
        self.MAX_N = MAX_N
        self.f_in_vert = f_in_vert
        self.f_in_edge = f_in_edge
        self.g1 = GraphEdgeVertLayer(f_in_vert, internal_d_vert, 
                                     f_in_edge, internal_d_edge)
        self.g_inner = nn.ModuleList([GraphEdgeVertLayer(internal_d_vert, 
                                                         internal_d_vert, 
                                                         internal_d_edge, 
                                                         internal_d_edge) for _ in range(layer_n)])
        #self.g3 = GraphEdgeVertLayer(internal_d_vert, 1, internal_d_edge, 1)
        
        self.as_resnet = resnet
        
        #self.bn_v_in = nn.BatchNorm1d(MAX_N * f_in_vert)
        #self.bn_v_inner = nn.ModuleList([nn.BatchNorm1d(MAX_N, internal_d_vert) for _ in range(layer_n)])

        self.outLin = nn.Linear(internal_d_vert * layer_n, 1)

        self.init_noise = INIT_NOISE
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.bias, -self.init_noise, self.init_noise)
                nn.init.uniform_(m.weight, -self.init_noise, self.init_noise)

    def forward(self, args):
        e, v = args

        e = torch.transpose(e, 1,3)
        
        #v = self.bn_v_in(v.view(-1, self.MAX_N * self.f_in_vert))
        #v = v.view(-1, self.MAX_N, self.f_in_vert)
        #print("v.shape=", v.shape, "e.shape=", e.shape)
        highway = []
        v, e = self.g1(v, e)
        for i in range(len(self.g_inner)):
        
            v_next, e_next = self.g_inner[i](v, e)
            if self.as_resnet:
                v = v_next + v
                e = e_next + e
            else:
                v = v_next
                e = e_next
            highway.append(v)
            #v = self.bn_v_inner[i](v)
        #v, e = self.g3(v, e)
        #print("v.shape=", v.shape, "e.shape=", e.shape)
        v_highway = torch.cat(highway, dim=-1) # , dim=-1)
        return self.outLin(v_highway)

