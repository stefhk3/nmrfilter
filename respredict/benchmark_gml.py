## Benchmarking GraphmatLayer



import graph_conv_many_nuc_util
import netdataio
import os
import torch
import nets as existing_nets
from graph_conv_many_nuc_util import move
from tqdm import tqdm_notebook as tqdm
import torch.nn as nn
import torch.nn.functional as F
import time

# from jax import device_put

# import jax
# import jax.numpy as np

# from jax.experimental import optimizers
# from jax import jit, grad, vmap


layer_n = 10
resnet = True
INT_D = 2048

agg_func = 'max'


MAT_CHAN_N = 4
GS = MAT_CHAN_N

MAX_N = 64


class GraphMatLayerReorder(nn.Module):
    def __init__(self, C, P , GS=1,  
                 noise=1e-6, agg_func=None, 
                 dropout=0.0):
        """
        Pairwise layer -- takes a N x M x M x C matrix
        and turns it into a N x M x M x P matrix after
        multiplying with a graph matrix N x M x M
        
        if GS != 1 then there will be a per-graph-channel 
        linear layer
        """
        super(GraphMatLayerReorder, self).__init__()

        self.GS = GS
        self.noise=noise

        self.linlayers = nn.ModuleList()
        self.dropout = dropout
        self.dropout_layers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P)
            l.bias.data.normal_(0.0, self.noise)
            l.weight.data.normal_(0.0, self.noise) #?!
            self.linlayers.append(l)
            if dropout > 0.0:
                self.dropout_layers.append(nn.Dropout(p=dropout))
            
        #self.r = nn.PReLU()
        self.r = nn.ReLU()
        self.agg_func = agg_func
 
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape
        def apply_ll(i, x):
            y = self.linlayers[i](x)
            return y

        multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)], 0)
        xout = torch.einsum("ijkl,jilm->jikm", [G, multi_x])

        if self.agg_func is not None:
            x = self.agg_func(xout, dim=0)
        return self.r(x)
        

class GraphMatLayerEinsum(nn.Module):
    def __init__(self, C, P , GS=1,  
                 noise=1e-6, agg_func=None, 
                 dropout=0.0):
        """
        Pairwise layer -- takes a N x M x M x C matrix
        and turns it into a N x M x M x P matrix after
        multiplying with a graph matrix N x M x M
        
        if GS != 1 then there will be a per-graph-channel 
        linear layer
        """
        super(GraphMatLayerEinsum, self).__init__()

        self.GS = GS
        self.noise=noise

        self.linlayers = nn.ModuleList()
        self.dropout = dropout
        self.dropout_layers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P)
            l.bias.data.normal_(0.0, self.noise)
            l.weight.data.normal_(0.0, self.noise) #?!
            self.linlayers.append(l)
            if dropout > 0.0:
                self.dropout_layers.append(nn.Dropout(p=dropout))
            
        #self.r = nn.PReLU()
        self.r = nn.ReLU()
        self.agg_func = agg_func
 
    def forward(self, G, x):
        BATCH_N, CHAN_N,  MAX_N, _ = G.shape
        def apply_ll(i, x):
            y = self.linlayers[i](x)
            if self.dropout > 0:
                y = self.dropout_layers[i](y)
            return y

        multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)])
        xout = torch.einsum("ijkl,jilm->ijkm", [G, multi_x])
        x = self.r(xout)
        if self.agg_func is not None:
            x = self.agg_func(x, dim=1)
        return x
        


class GraphMatLayerJIT(torch.jit.ScriptModule):
    __constants__ = ['GS', 'linlayers', 'agg_func']

    def __init__(self, C, P , GS=1,  
                 noise=1e-6, agg_func=None, 
                 dropout=0.0):
        """
        Pairwise layer -- takes a N x M x M x C matrix
        and turns it into a N x M x M x P matrix after
        multiplying with a graph matrix N x M x M
        
        if GS != 1 then there will be a per-graph-channel 
        linear layer
        """
        super(GraphMatLayerJIT, self).__init__()

        self.GS = GS
        self.noise=noise


        self.dropout = dropout
        self.dropout_layers = nn.ModuleList()
        linlayers = []
        for ll in range(GS):
            l = nn.Linear(C, P)
            l.bias.data.normal_(0.0, self.noise)
            l.weight.data.normal_(0.0, self.noise) #?!
            linlayers.append(l)
        self.linlayers = nn.ModuleList(linlayers)            
        #self.r = nn.PReLU()
        self.r = nn.ReLU()
        self.agg_func = agg_func

    @torch.jit.script_method
    def forward(self, G, x):
        
        a = []
        for l in self.linlayers:
            a.append(l(x))
        multi_x = torch.stack(a)
        #multi_x = torch.stack([self.linlayers[i](x) for i in range(self.GS)])

        xout = []
        for i in range(G.shape[0]):
            xout.append(torch.matmul(G[i], multi_x[:, i]))
        xout = torch.stack(xout)

        x = self.r(xout)

        x = torch.max(x, dim=1)[0]
        # if self.agg_func is not None:
        #     x = self.agg_func(x, dim=1)
        return x
        

class GraphMatLayersCustom(torch.jit.ScriptModule):
    __constants__ = ['gl', 'resnet']
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 noise=1e-5, agg_func=None, dropout=0.0, 
                 gml_class=existing_nets.GraphMatLayer):
        super(GraphMatLayersCustom, self).__init__()
        

        self.resnet = resnet
        gl_list = []
        for li in range(len(output_features_n)):
            if li == 0:
                gl = gml_class(input_feature_n, output_features_n[0],
                               noise=noise, agg_func=agg_func, GS=GS, 
                                   dropout=dropout)
            else:
                gl = gml_class(output_features_n[li-1], 
                                   output_features_n[li], 
                                   noise=noise, agg_func=agg_func, GS=GS, 
                                   dropout=dropout)
            
            gl_list.append(gl)
        self.gl = nn.ModuleList(gl_list)
        
    #@torch.jit.script_method
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



def pytorch_benchmark():



    USE_CUDA = True


    BATCH_N = 16

    input_data = move(torch.zeros((BATCH_N, MAX_N, INT_D)), USE_CUDA)
    G_data = move(torch.zeros((BATCH_N, MAT_CHAN_N, MAX_N, MAX_N)), USE_CUDA)

    # gml = GraphMatLayersCustom(INT_D, [INT_D]*layer_n, resnet=resnet, GS=GS, 
    #                            agg_func = existing_nets.goodmax, 
    #                            gml_class=GraphMatLayerReorder)
    gml = existing_nets.GraphMatLayers(INT_D, [INT_D]*layer_n, resnet=resnet, GS=GS, 
                         agg_func = existing_nets.goodmax)
                         
    gml = move(gml, USE_CUDA)

    total = 0.0

    ITERS = 100

    t1 = time.time()
    for i in range(ITERS):
        r = gml(G_data, input_data)
        y = r.sum()
        total += y.detach().cpu().numpy()
    t2 =time.time()
    print("{:3.1f} points/sec".format(BATCH_N*ITERS/(t2-t1)))


# @jit
# def foo(G_in, x, param_W, param_b):
#     y1 = np.dot(x, param_W) + param_b
#     y = np.dot(G_in, y1)
#     return y

# @jit
# def layer(G_in, x, param_W, param_b):

#     """
#     G_in is (MAT_C x MAX_N x MAX_N)
#     x is (MAX_N x INT_D)
#     param_W is MATRIX_C x INT_D x INT_D
#     param_b is MATIRX_C x INT_D 

#     """
#     # C = 4 # G_in.shape[0]
#     # multi_x = []
#     # for c in range(C):
#     #     y = np.dot(x, param_W[c]) + param_b[c] 
#     #     y = np.dot(G_in[c], y)
#     #     multi_x.append(y)

#     # stacked = np.stack(multi_x, -1)

#     stacked = vmap(foo, in_axes=(0, None, 0, 0), out_axes=2, )(G_in, x, param_W, param_b)
#     return np.maximum(np.max(stacked, axis=2), 0.0)



# @jit
# def layer_batch(G_in, x, param_W, param_b):

#     """
#     G_in is (BATCH_N, MAT_C x MAX_N x MAX_N)
#     x is (BATCH_N, MAX_N x INT_D)
#     param_W is MATRIX_C x INT_D x INT_D
#     param_b is MATIRX_C x INT_D 

#     """
#     def _lin_layer(x, W, b):
#         return np.dot(x, W) + b
    
#     lin_layer = vmap(_lin_layer, in_axes=(0, None, None))

#     # def apply_ll(i, x_in):
#     #     return lin_layer(x_in, param_W[i], param_b[i])

#     # multi_x = np.stack([apply_ll(i,x) for i in range(G_in.shape[1])])

#     def apply_ll(i, W, b):
#         return lin_layer(x_in, W, b)

#     multi_x = vmap(lin_layer, in_axes=(None, 0, 0))(x, param_W, param_b)
#     # scan over batch
#     #xout = np.stack([np.matmul(G_in[i], multi_x[:, i]) for i in range(x.shape[0])])
#     xout = vmap(np.matmul, in_axes=(0, 1))(G_in, multi_x) 
#     xout = np.maximum(xout, 0)
#     return np.max(xout, axis=1)

# @jit
# def multi_layer(G_in, x, param_W, param_b):
#     for layer_i in range(param_W.shape[0]):

#         new_x = layer(G_in, x, param_W[layer_i], param_b[layer_i])
#         x = new_x + x 
#     return x

# def jax_benchmark():


#     BATCH_N = 16
#     MAX_N = 64

#     USE_CUDA = False

#     input_data = np.zeros((BATCH_N, MAX_N, INT_D))
#     G_data = np.zeros((BATCH_N, MAT_CHAN_N, MAX_N, MAX_N))

#     vmap_layer = vmap(layer, in_axes=(0, 0, None, None))

#     param_W = device_put(np.zeros((layer_n, MAT_CHAN_N, INT_D, INT_D)))
#     param_b = device_put(np.zeros((layer_n, MAT_CHAN_N, INT_D)))
#     input_data = device_put(input_data)
#     G_data = device_put(G_data)

#     vmap_multi_layer = vmap(multi_layer, in_axes=(0, 0, None, None))
#     #vmap_multi_layer = vmap(multi_layer, in_axes=(0, 0, None, None))
#     print("the jax is")
#     print(vmap_multi_layer)
#     total = 0.0

#     ITERS = 50

#     t1 = time.time()
#     for i in range(ITERS):
#         r = vmap_multi_layer(G_data, input_data, param_W, param_b)
#         y = r.sum()
#         total += y
#     t2 =time.time()
#     print("{:3.1f} datapoints/sec".format((ITERS*BATCH_N)/(t2-t1)))


# @jit
# def multi_layer_batch(G_in, x, param_W, param_b):
#     for layer_i in range(param_W.shape[0]):

#         new_x = layer_batch(G_in, x, param_W[layer_i], param_b[layer_i])
#         x = new_x + x
#     return x


# def jax_benchmark_2():


#     BATCH_N = 16
#     MAX_N = 64

#     USE_CUDA = False

#     input_data = np.zeros((BATCH_N, MAX_N, INT_D))
#     G_data = np.zeros((BATCH_N, MAT_CHAN_N, MAX_N, MAX_N))

#     vmap_layer = vmap(layer, in_axes=(0, 0, None, None))

#     param_W = device_put(np.zeros((layer_n, MAT_CHAN_N, INT_D, INT_D)))
#     param_b = device_put(np.zeros((layer_n, MAT_CHAN_N, INT_D)))
#     input_data = device_put(input_data)
#     G_data = device_put(G_data)

#     #vmap_multi_layer = vmap(multi_layer, in_axes=(0, 0, None, None))
#     #vmap_multi_layer = vmap(multi_layer, in_axes=(0, 0, None, None))

#     total = 0.0

#     ITERS = 100

#     t1 = time.time()
#     for i in range(ITERS):
#         r = multi_layer_batch(G_data, input_data, param_W, param_b)
#         #y = r.sum()
#         #total += y
#     t2 =time.time()
#     print("{:3.1f} datapoints/sec".format((ITERS*BATCH_N)/(t2-t1)))


if __name__ == "__main__":
    pytorch_benchmark()
    #jax_benchmark_2()
