# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 00:04:23 2024

@author: naisops
"""

import torch
from torch import nn
import torch.nn.functional as F
import math
import einops
import copy

device = torch.device('cuda:0') 
drop_out=0.1

class PositionalEncoder(nn.Module):
    # output shape: (bs, seq_length, (p * p))
    def __init__(self, d_model, seq_len, dropout = drop_out):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is in the shape of (bs, seq_length, (p * p))
        x = x + self.pos_embed
        return self.dropout(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = drop_out):
        super().__init__()
        
        self.d_model = d_model
        self.d_head = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    
    def forward(self, q, k, v):
        bs = q.size(0)
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_head)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_head)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_head)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores = attention(q, k, v, self.d_head, self.dropout)
        # scores shape: (bs x h x sl x d_k)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)
        # output shape: (bs x sl x d_model)
    
        return output
       
def attention(q, k, v, d_k, dropout=None):  
    # q,k,v shape: (bs x h x sl x d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    # output shape: (bs x h x sl x d_k)
    return output

class FeedForward(nn.Module):
    
    def __init__(self, d_model, dropout = drop_out):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, d_model*2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_model*2, d_model)
        
    def forward(self, x):
        x = self.dropout(self.act(self.linear_1(x)))
        x = self.linear_2(x)
        
        return x
    
class Norm(nn.Module):
    
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        norm = self.norm(x)
        
        return norm
    
class EncoderLayer(nn.Module):
    # inpuut: (bs,C,H,W)
    
    def __init__(self, d_model, heads, dropout = drop_out):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x2 = self.norm_1(x)
        temp = self.attn(x2,x2,x2)
        x = x + self.dropout_1(temp)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class transBlock(nn.Module):
    def __init__(self, d, heads, n_trans):
        super().__init__()
        layers = []
        self.n_trans = n_trans
        for i in range(n_trans):
            layers.append(EncoderLayer(d, heads))
             
        self.sequential = nn.Sequential(*layers)
    
    def forward(self, x):
        for i in range(self.n_trans):
            x = self.sequential[i](x)
            
        return x
    
    
class ConvBlock(nn.Module):
    # input (bs, C_in, D_in, H_in, W_in)
    # output (bs, C_out, D_out, H_out, W_out)
    
    def __init__(self, in_channels, out_channels, padding = (1,1), kernel_size = 3, dropout=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding = padding)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.dropout = dropout
        if dropout:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding = padding)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding = padding),
            nn.BatchNorm2d(out_channels),
        )

        
    def forward(self, x):
        skip = self.conv_skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        if self.dropout:
            x = self.dropout1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        if self.dropout:
            x = self.dropout2(x)

        return x+skip
    
    
### Base_single & Base_multi###
class CTS(nn.Module):

    def __init__(self, ini_channel, base_channel, convN, im_size, n_trans, num_class,c0):
        super().__init__()
        self.num_class = num_class
        self.im_size = im_size
        self.convN = convN
        self.n = im_size // (2**convN)
        p = 2**convN
        d = base_channel*(2**(convN))//c0
        self.p = p

        self.conv = []
        self.conv.append(ConvBlock(ini_channel, base_channel))
        for i in range(convN):
            self.conv.append(nn.MaxPool2d(2))
            self.conv.append(ConvBlock(base_channel*(2**(i)), base_channel*(2**(i+1))))
        self.conv = nn.Sequential(*self.conv)
        
        self.linear = []
        for i in range(convN+1):
            self.linear.append(nn.Linear(base_channel*(2**(i)),base_channel*(2**(i))//c0))
        self.linear = nn.Sequential(*self.linear)
        
        self.pe = PositionalEncoder(d, (im_size//p)**2)
        self.trans = []
        self.trans0 = transBlock(d, d//(base_channel), n_trans)

        for i in range(convN):
            self.trans.append(nn.Linear(d*(2**i),d*(2**(i+1))))
            self.trans.append(transBlock(d*(2**(i+1)), d*(2**(i+1))//base_channel, n_trans)) 
        self.trans = nn.Sequential(*self.trans)

        self.seg_head = nn.Conv2d(base_channel//c0, num_class, (1,1))
        if num_class < 2:
            self.out = nn.Sigmoid()
        
        
    def forward(self, x, convn=True): 
        cnn_out = []
        for i,l in enumerate(self.conv):
            x = l(x)
            if i%2 == 0:
                x0 = einops.rearrange(x, 'b c h w -> b h w c')
                x0 = self.linear[i//2](x0)
                x0 = einops.rearrange(x0, 'b (n1 p1) (n2 p2) c -> b (c p1 p2) n1 n2', n1=self.n, n2=self.n)
                x0 = einops.rearrange(x0, 'b c h w -> b (h w) c')
                cnn_out.append(x0)
        
        x = cnn_out[-1]
        x = self.pe(x)
        x = self.trans0(x)

        for i,l in enumerate(self.trans):
            if i % 2 == 0:
                x = l(x)
                x = x + (cnn_out[self.convN - 1 - i//2])
            else:
                x = l(x)
        
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=self.n, w=self.n)
        x = einops.rearrange(x, 'b (c p1 p2) n1 n2 -> b c (n1 p1) (n2 p2)', p1=2**self.convN, p2=2**self.convN)
        x = self.seg_head(x)
        if self.num_class < 2:
            x = self.out(x)
        
        return x
    
### Decoder_multi ###
class CTS_multi(nn.Module):

    def __init__(self, ini_channel, base_channel, convN, im_size, n_trans, num_class,c0,n_task):
        super().__init__()
        self.num_class = num_class
        self.im_size = im_size
        self.convN = convN
        self.n = im_size // (2**convN)
        p = 2**convN
        d = base_channel*(2**(convN))//c0
        self.p = p

        self.conv = []
        self.conv.append(ConvBlock(ini_channel, base_channel))
        for i in range(convN):
            self.conv.append(nn.MaxPool2d(2))
            self.conv.append(ConvBlock(base_channel*(2**(i)), base_channel*(2**(i+1))))
        self.conv = nn.Sequential(*self.conv)
        
        self.linear = []
        for i in range(convN+1):
            self.linear.append(nn.Linear(base_channel*(2**(i)),base_channel*(2**(i))//c0))
        self.linear = nn.Sequential(*self.linear)
        
        self.pe = PositionalEncoder(d, (im_size//p)**2)
        self.trans = nn.ModuleList()
        self.seg_head = nn.ModuleList()
        
        for t in range(n_task):
            trans_block_list = []
            trans_block_list.append(transBlock(d, d//(base_channel), n_trans))
            for i in range(convN):
                trans_block_list.append(nn.Linear(d*(2**i),d*(2**(i+1))))
                trans_block_list.append(transBlock(d*(2**(i+1)), d*(2**(i+1))//base_channel, n_trans)) 
            self.trans.append(nn.Sequential(*trans_block_list))

            self.seg_head.append(nn.Conv2d(base_channel//c0, num_class, (1,1)))
            
        if num_class < 2:
            self.out = nn.Sigmoid()
            
    def decoder(self,x,cnn_out,t,b):
        for i,l in enumerate(self.trans[t]):
            if i % 2 == 1:
                x = l(x)
                x = x + (cnn_out[self.convN - 1 - i//2][b].unsqueeze(0))
            else:
                x = l(x)
        
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=self.n, w=self.n)
        x = einops.rearrange(x, 'b (c p1 p2) n1 n2 -> b c (n1 p1) (n2 p2)', p1=2**self.convN, p2=2**self.convN)
        x = self.seg_head[t](x)
        
        return x
        
    def forward(self, x,token, convn=True): 
        cnn_out = []
        for i,l in enumerate(self.conv):
            x = l(x)
            if i%2 == 0:
                x0 = einops.rearrange(x, 'b c h w -> b h w c')
                x0 = self.linear[i//2](x0)
                x0 = einops.rearrange(x0, 'b (n1 p1) (n2 p2) c -> b (c p1 p2) n1 n2', n1=self.n, n2=self.n)
                x0 = einops.rearrange(x0, 'b c h w -> b (h w) c')
                cnn_out.append(x0)
        
        x = cnn_out[-1]
        x = self.pe(x)
        
        x_out = []
        for i in range(x.size(0)):
            ct = token[i]['CLASS']
            vt = token[i]['VIEW']
            if ct == 0 and vt == 0:
                x_out.append(self.decoder(x[i].unsqueeze(0),cnn_out,0,i))
            elif ct == 1 and vt == 0:
                x_out.append(self.decoder(x[i].unsqueeze(0),cnn_out,1,i))
            elif ct == 1 and vt == 1:
                x_out.append(self.decoder(x[i].unsqueeze(0),cnn_out,2,i))
            elif ct == 2 and vt == 1:
                x_out.append(self.decoder(x[i].unsqueeze(0),cnn_out,3,i))
        
        x_out = torch.cat(x_out,dim=0)
               
        if self.num_class < 2:
            x_out = self.out(x_out)
        
        return x_out
    
### proposed network ###
class CTS_cond(nn.Module):

    def __init__(self, ini_channel, base_channel, convN, im_size, n_trans, num_class,c0,n_task=4,learnable=True):
        super().__init__()
        self.num_class = num_class
        self.im_size = im_size
        self.convN = convN
        self.n = im_size // (2**convN)
        p = 2**convN
        d = base_channel*(2**(convN))//c0
        self.p = p

        self.conv = []
        self.conv.append(ConvBlock(ini_channel, base_channel))
        for i in range(convN):
            self.conv.append(nn.MaxPool2d(2))
            self.conv.append(ConvBlock(base_channel*(2**(i)), base_channel*(2**(i+1))))
        self.conv = nn.Sequential(*self.conv)
        
        self.linear = []
        for i in range(convN+1):
            self.linear.append(nn.Linear(base_channel*(2**(i)),base_channel*(2**(i))//c0))
        self.linear = nn.Sequential(*self.linear)
        
        self.class_embb = []
        for i in range(n_task):
            if learnable:
                self.class_embb.append(nn.Parameter(torch.randn(1, 1, d)))
            else:
                self.class_embb.append(nn.Parameter(torch.ones(1, 1, d)*(i+1)))
        self.class_embb = nn.ParameterList(self.class_embb)
        
        self.pe = PositionalEncoder(d, 1+(im_size//p)**2)
        self.trans = []
        self.trans0 = transBlock(d, d//(base_channel), n_trans)

        for i in range(convN):
            self.trans.append(nn.Linear(d*(2**i),d*(2**(i+1))))
            self.trans.append(transBlock(d*(2**(i+1)), d*(2**(i+1))//base_channel, n_trans)) 
        self.trans = nn.Sequential(*self.trans)

        self.seg_head = nn.Conv2d(base_channel//c0, num_class, (1,1))
        if num_class < 2:
            self.out = nn.Sigmoid()
        
        
    def forward(self, x, token, convn=True): 
        cnn_out = []
        for i,l in enumerate(self.conv):
            x = l(x)
            if i%2 == 0:
                x0 = einops.rearrange(x, 'b c h w -> b h w c')
                x0 = self.linear[i//2](x0)
                x0 = einops.rearrange(x0, 'b (n1 p1) (n2 p2) c -> b (c p1 p2) n1 n2', n1=self.n, n2=self.n)
                x0 = einops.rearrange(x0, 'b c h w -> b (h w) c')
                cnn_out.append(x0)
        
        x = cnn_out[-1]
        
        x_out = []
        for i in range(x.size(0)):
            ct = token[i]['CLASS']
            vt = token[i]['VIEW']
            if ct == 0 and vt == 0:
                x_out.append(torch.cat((x[i].unsqueeze(0),self.class_embb[0]),dim=1))
            elif ct == 1 and vt == 0:
                x_out.append(torch.cat((x[i].unsqueeze(0),self.class_embb[1]),dim=1))
            elif ct == 1 and vt == 1:
                x_out.append(torch.cat((x[i].unsqueeze(0),self.class_embb[2]),dim=1))
            elif ct == 2 and vt == 1:
                x_out.append(torch.cat((x[i].unsqueeze(0),self.class_embb[3]),dim=1))
        x_out = torch.cat(x_out,dim=0)      
                
        x = self.pe(x_out)
        x = self.trans0(x)

        for i,l in enumerate(self.trans):
            if i % 2 == 0:
                x = l(x)
                x[:,:-1] = x[:,:-1] + (cnn_out[self.convN - 1 - i//2])
            else:
                x = l(x)
        
        x = x[:,:-1]
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=self.n, w=self.n)
        x = einops.rearrange(x, 'b (c p1 p2) n1 n2 -> b c (n1 p1) (n2 p2)', p1=2**self.convN, p2=2**self.convN)
        x = self.seg_head(x)
        if self.num_class < 2:
            x = self.out(x)
        
        return x
    
class FeedForward_dyna(nn.Module):
    
    def __init__(self, d_model, d_out, dropout = drop_out):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_model, d_out)
        
    def forward(self, x):
        x = self.dropout(self.act(self.linear_1(x)))
        x = self.linear_2(x)
        
        return x
    
class DecoderLayer(nn.Module):
    # inpuut: (bs,C,H,W)
    
    def __init__(self, d_model, heads,d_dyna, dropout = drop_out):
        super().__init__()
        self.norm_p1 = Norm(d_model)
        self.norm_p2 = Norm(d_model)
        self.norm_p3 = Norm(d_model)
        # self.norm_p4 = Norm(d_model)
        self.norm_i1 = Norm(d_model)
        # self.norm_i2 = Norm(d_model)
        self.attn_pp = MultiHeadAttention(heads, d_model)
        self.attn_pi = MultiHeadAttention(heads, d_model)
        # self.attn_ip = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward_dyna(d_model,d_dyna)
        self.dropout_p1 = nn.Dropout(dropout)
        self.dropout_p2 = nn.Dropout(dropout)
        # self.dropout_p3 = nn.Dropout(dropout)
        # self.dropout_i1 = nn.Dropout(dropout)
        
    def forward(self, image, prompt, posi, posp):
        prompt0 = prompt+posp
        xp = self.norm_p1(prompt+posp)
        xp = self.attn_pp(xp,xp,xp)
        prompt = prompt + self.dropout_p1(xp)
        xp = self.norm_p2(prompt+prompt0)
        xi = self.norm_i1(image+posi)
        xp = self.attn_pi(xp,xi,xi)
        prompt = prompt + self.dropout_p2(xp)
        xp = self.norm_p3(prompt+prompt0)
        xp = self.ff(xp)
        # prompt = prompt + self.dropout_p3(xp)
        # xi = self.norm_i2(image+posi)
        # xp = self.norm_p4(prompt+prompt0)
        # xi = self.attn_ip(xi,xp,xp)
        # image = image + self.dropout_i1(xi)
        
        return xp
    
    
### Decoder_dynamic ###
class CTS_dynamic(nn.Module):

    def __init__(self, ini_channel, base_channel, convN, im_size, n_trans, num_class,c0,n_task=4):
        super().__init__()
        self.num_class = num_class
        self.im_size = im_size
        self.convN = convN
        self.n = im_size // (2**convN)
        p = 2**convN
        d = base_channel*(2**(convN))//c0
        self.p = p
        
        self.final_c = base_channel//c0
        dy_weights_num = (self.final_c**2+self.final_c)*2 + self.final_c*num_class+num_class

        self.conv = []
        self.conv.append(ConvBlock(ini_channel, base_channel))
        for i in range(convN):
            self.conv.append(nn.MaxPool2d(2))
            self.conv.append(ConvBlock(base_channel*(2**(i)), base_channel*(2**(i+1))))
        self.conv = nn.Sequential(*self.conv)
        
        self.linear = []
        for i in range(convN+1):
            self.linear.append(nn.Linear(base_channel*(2**(i)),base_channel*(2**(i))//c0))
        self.linear = nn.Sequential(*self.linear)
        
        self.pt = nn.Parameter(torch.randn(1, n_task, d))
        self.posi = nn.Parameter(torch.randn(1, (im_size//p)**2, d))
        self.posp = nn.Parameter(torch.randn(1, n_task, d))
        
        self.trans = []
        self.trans0 = transBlock(d, d//(base_channel), n_trans)
        
        self.dec = DecoderLayer(d, d//(base_channel),dy_weights_num)

        for i in range(convN):
            self.trans.append(nn.Linear(d*(2**i),d*(2**(i+1))))
            self.trans.append(transBlock(d*(2**(i+1)), d*(2**(i+1))//base_channel, n_trans)) 
        self.trans = nn.Sequential(*self.trans)

        if num_class < 2:
            self.out = nn.Sigmoid()
            
    def split_weights_biases(self,embb,final_c,num_class):
        start = 0
        w1 = embb[:,start:start+final_c**2].reshape(final_c,final_c,1,1)
        start += final_c**2
        b1 = embb[:,start:start+final_c].flatten()
        start += final_c
        w2 = embb[:,start:start+final_c**2].reshape(final_c,final_c,1,1)
        start += final_c**2
        b2 = embb[:,start:start+final_c].flatten()
        start += final_c
        w3 = embb[:,start:start+final_c*num_class].reshape(num_class,final_c,1,1)
        start += final_c*num_class
        b3 = embb[:,start:start+num_class].flatten()
        
        return (w1,w2,w3), (b1,b2,b3)
            
    def heads_forward(self, features, weights, biases):
        
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x
        
        
    def forward(self, x, token, convn=True): 
        cnn_out = []
        for i,l in enumerate(self.conv):
            x = l(x)
            if i%2 == 0:
                x0 = einops.rearrange(x, 'b c h w -> b h w c')
                x0 = self.linear[i//2](x0)
                x0 = einops.rearrange(x0, 'b (n1 p1) (n2 p2) c -> b (c p1 p2) n1 n2', n1=self.n, n2=self.n)
                x0 = einops.rearrange(x0, 'b c h w -> b (h w) c')
                cnn_out.append(x0)
        
        x = cnn_out[-1]
        x = x + self.posi
        x = self.trans0(x)
        x1 = x

        for i,l in enumerate(self.trans):
            if i % 2 == 0:
                x = l(x)
                x = x + (cnn_out[self.convN - 1 - i//2])
            else:
                x = l(x)
        
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=self.n, w=self.n)
        x = einops.rearrange(x, 'b (c p1 p2) n1 n2 -> b c (n1 p1) (n2 p2)', p1=2**self.convN, p2=2**self.convN)
        
        x_out = []
        for i in range(x.size(0)):
            pt = self.pt
            pt = self.dec(x1[i].unsqueeze(0), pt, self.posi, self.posp)
            
            ct = token[i]['CLASS']
            vt = token[i]['VIEW']
            if ct == 0 and vt == 0:
                ws,bs = self.split_weights_biases(pt[:,0],self.final_c,self.num_class)
            elif ct == 1 and vt == 0:
                ws,bs = self.split_weights_biases(pt[:,1],self.final_c,self.num_class)
            elif ct == 1 and vt == 1:
                ws,bs = self.split_weights_biases(pt[:,2],self.final_c,self.num_class)
            elif ct == 2 and vt == 1:
                ws,bs = self.split_weights_biases(pt[:,3],self.final_c,self.num_class)
                
            x_out.append(self.heads_forward(x[i].unsqueeze(0),ws,bs))
            
        x_out = torch.cat(x_out,dim=0)
        if self.num_class < 2:
            x_out = self.out(x_out)
        
        return x_out
    

