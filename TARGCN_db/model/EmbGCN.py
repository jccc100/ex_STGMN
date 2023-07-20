import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


device=torch.device('cuda')
def sym_norm_Adj(W):

    W=W.to(device=torch.device('cpu'))
    assert W.shape[0] == W.shape[1]
    W=W.cpu().detach().numpy()

    N = W.shape[0]

    W = W + 0.5*np.identity(N) # 为邻居矩阵加上自连接

    D = np.diag(1.0/np.sum(W, axis=1))

    sym_norm_Adj_matrix = np.dot(np.sqrt(D),W)

    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix,np.sqrt(D))

    return sym_norm_Adj_matrix # D^-0.5AD^-0.5
class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, num_node,c_in,c_out,dropout=.0):
        super(Spatial_Attention_layer, self).__init__()
        global device
        self.in_channels=c_in
        self.dropout = nn.Dropout(p=dropout)

        # self.conv1 = nn.Conv2d(num_node, num_node, (1, 3), bias=False)
        # self.conv2 = nn.Conv2d(num_node, num_node, (1, 3), bias=True)
        self.Wq=nn.Linear(c_in,c_out,bias=True)
        # nn.init.kaiming_uniform_(self.Wq.weight, nonlinearity="relu")
        self.Wk=nn.Linear(c_in,c_out,bias=True)
        # nn.init.kaiming_uniform_(self.Wk.weight, nonlinearity="relu")
        self.Wv=nn.Linear(c_in,c_out,bias=False)
        # # nn.init.kaiming_uniform_(self.Wv.weight, nonlinearity="relu")
    def forward(self, x,adj,score_his=None):
        '''
        :param x: (batch_size, N, C)
        :return: (batch_size, N, C)
        '''
        batch_size, num_of_vertices, in_channels = x.shape

        Q=self.Wq(x)
        K=self.Wk(x)
        V=self.Wv(x)
        score = torch.matmul(Q, K.transpose(1, 2))
        score=F.softmax(score,dim=1)
        score=torch.einsum('bnm,mc->bnc',score,adj)
        score=torch.einsum("bnm,bmc->bnc",score,V)
        return score # (b n n)
class EmbGCN(nn.Module):
    def __init__(self, dim_in, dim_out, adj,cheb_k, embed_dim):
        super(EmbGCN, self).__init__()
        self.cheb_k = cheb_k
        self.sym_norm_Adj_matrix = torch.from_numpy(sym_norm_Adj(adj)).to(torch.float32).to(torch.device('cuda'))
        self.sym_norm_Adj_matrix=F.softmax(self.sym_norm_Adj_matrix)
        self.linear=nn.Linear(dim_in, dim_out,bias=True)

        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]

        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1) # N N
        supports = torch.eye(node_num).to(supports.device)+supports

        # 静态邻接矩阵
        x_static = torch.einsum("nm,bmc->bmc",torch.softmax(self.sym_norm_Adj_matrix,dim=-1),x)
        x_static = self.linear(x_static)

        weights = torch.einsum('nd,dio->nio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)#N, dim_out

        x_g = torch.einsum("nm,bmc->bnc", supports, x)      #B, cheb_k, N, dim_in

        # x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bni,nio->bno', x_g, weights) + bias     #b, N, dim_out
        # return x_gconv
        return x_gconv+torch.sigmoid(x_static)*x_static

# class EmbGCN_test(nn.Module):
#     def __init__(self, dim_in, dim_out, adj,cheb_k, embed_dim):
#         super(EmbGCN_test, self).__init__()
#         self.cheb_k = cheb_k
#         self.sym_norm_Adj_matrix = torch.from_numpy(sym_norm_Adj(adj)).to(torch.float32).to(torch.device('cuda'))
#         self.sym_norm_Adj_matrix=F.softmax(self.sym_norm_Adj_matrix)
#         self.linear=nn.Linear(dim_in, dim_out,bias=True)
#         self.linear2=nn.Linear(dim_in, dim_out,bias=True)
#         # self.sym_norm_Adj_matrix=F.softmax(torch.Tensor(adj).to(torch.device('cuda')))
#         # self.SA=Spatial_Attention_layer(adj.shape[0],dim_in,dim_out)
#         # self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
#         # self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
#
#         self.A=nn.Parameter(torch.FloatTensor(170, 170),requires_grad=True)
#     def forward(self, x, node_embeddings):
#         #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
#         #output shape [B, N, C]
#         # x=self.SA(x)
#         node_num = node_embeddings.shape[0]
#         supports = F.softmax(F.relu(torch.mm(self.A, self.A.transpose(0, 1))), dim=1) # N N
#         supports = torch.eye(node_num).to(supports.device)+supports
#         # support_set = [torch.eye(node_num).to(supports.device), supports]
#         # supports = torch.stack(support_set, dim=0)
#         # 静态邻接矩阵
#         x_static = torch.einsum("nm,bmc->bmc",torch.softmax(self.sym_norm_Adj_matrix,dim=-1),x)
#         x_static = self.linear(x_static)
#         #
#         # x_static = self.SA(x,self.sym_norm_Adj_matrix)
#         # x_static=F.relu(x_static)
#
#
#
#         # weights = torch.einsum('nd,dio->nio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
#         # bias = torch.matmul(node_embeddings, self.bias_pool)#N, dim_out
#         # supports=torch.einsum('knm,mm->knm',supports,self.sym_norm_Adj_matrix)
#         #加空注
#
#         x_g = torch.einsum("nm,bmc->bnc", supports, x)      #B, cheb_k, N, dim_in
#
#         # x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
#         # x_gconv = torch.einsum('bni,nio->bno', x_g, weights) + bias     #b, N, dim_out
#         x_gconv = self.linear2(x_g)
#         # return x_gconv
#         return x_gconv+torch.sigmoid(x_static)*x_static
#         # return x_gconv+(torch.tanh(x_gconv)+torch.sigmoid(x_static))
#
#         # return x_gconv+torch.sigmoid(x_static)*x_static+torch.sigmoid(x_gconv)*x_gconv
#         # return torch.sigmoid(x_static)*x_static+(1-torch.sigmoid(x_static))*x_gconv
class EmbGCN_noGate(nn.Module):
    def __init__(self, dim_in, dim_out, adj,cheb_k, embed_dim):
        super(EmbGCN_noGate, self).__init__()
        self.cheb_k = cheb_k
        self.cheb_k = cheb_k
        self.sym_norm_Adj_matrix = torch.from_numpy(sym_norm_Adj(adj)).to(torch.float32).to(torch.device('cuda'))
        self.sym_norm_Adj_matrix = F.softmax(self.sym_norm_Adj_matrix)
        self.linear = nn.Linear(dim_in, dim_out, bias=True)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        # x=self.SA(x)
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)  # N N
        supports = torch.eye(node_num).to(supports.device) + supports
        # support_set = [torch.eye(node_num).to(supports.device), supports]
        # supports = torch.stack(support_set, dim=0)
        # 静态邻接矩阵
        x_static = torch.einsum("nm,bmc->bmc", torch.softmax(self.sym_norm_Adj_matrix, dim=-1), x)
        x_static = self.linear(x_static)
        #
        # x_static = self.SA(x,self.sym_norm_Adj_matrix)
        # x_static=F.relu(x_static)

        weights = torch.einsum('nd,dio->nio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        # supports=torch.einsum('knm,mm->knm',supports,self.sym_norm_Adj_matrix)
        # 加空注

        x_g = torch.einsum("nm,bmc->bnc", supports, x)  # B, cheb_k, N, dim_in

        # x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bni,nio->bno', x_g, weights) + bias  # b, N, dim_out
        # return x_gconv
        return x_gconv + x_static


class EmbGCN_noPreA(nn.Module):
    def __init__(self, dim_in, dim_out, adj,cheb_k, embed_dim):
        super(EmbGCN_noPreA, self).__init__()
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        # x=self.SA(x)
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1) # N N
        supports = torch.eye(node_num).to(supports.device)+supports



        weights = torch.einsum('nd,dio->nio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)#N, dim_out
        # supports=torch.einsum('knm,mm->knm',supports,self.sym_norm_Adj_matrix)

        x_g = torch.einsum("nm,bmc->bnc", supports, x)      #B, cheb_k, N, dim_in

        # x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bni,nio->bno', x_g, weights) + bias     #b, N, dim_out
        # return x_gconv
        return x_gconv

class EmbGCN_linear(nn.Module):
    def __init__(self, dim_in, dim_out, adj,cheb_k, embed_dim):
        super(EmbGCN_linear, self).__init__()
        self.cheb_k = cheb_k
        # self.sym_norm_Adj_matrix = torch.from_numpy(sym_norm_Adj(adj)).to(torch.float32).to(torch.device('cuda'))
        # self.sym_norm_Adj_matrix=F.softmax(self.sym_norm_Adj_matrix)
        self.linear=nn.Linear(dim_in, dim_out,bias=True)
        # self.sym_norm_Adj_matrix=F.softmax(torch.Tensor(adj).to(torch.device('cuda')))
        # self.SA=Spatial_Attention_layer(adj.shape[0],dim_in,dim_out)
        # self.linear_weight=nn.Parameter(torch.FloatTensor(170,cheb_k,dim_in,dim_out),requires_grad=True)
        # self.linear_b=nn.Parameter(torch.FloatTensor(170,dim_out),requires_grad=True)
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        # x=self.SA(x)
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1) # N N
        # support_set = [torch.eye(node_num).to(supports.device), supports]
        # supports = torch.stack(support_set, dim=0)
        supports=torch.eye(node_num).to(supports.device)+supports # 1+A
        # print("supports:",supports.shape) #2 170 170
        # 静态邻接矩阵
        # x_static = torch.einsum("nm,bmc->bmc",torch.softmax(self.sym_norm_Adj_matrix,dim=-1),x)
        # x_static = self.linear(x_static)
        #
        # x_static = self.SA(x,self.sym_norm_Adj_matrix)
        # x_static=F.relu(x_static)



        # weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        # bias = torch.matmul(node_embeddings, self.bias_pool)#N, dim_out
        # supports=torch.einsum('knm,mm->knm',supports,self.sym_norm_Adj_matrix)
        #加空注

        x_g = torch.einsum("nm,bmc->bnc", supports, x)      #B, N, dim_in

        # x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # TARGCN_linear
        # x_gconv=torch.einsum('bnki,nkio->bno',x_g,self.linear_weight)+self.linear_b
        x_gconv=self.linear(x_g)
        return x_gconv
        # return x_gconv+(torch.tanh(x_gconv)+torch.sigmoid(x_static))
        # return x_gconv+torch.sigmoid(x_static)*x_static
        # return x_gconv+torch.sigmoid(x_static)*x_static+torch.sigmoid(x_gconv)*x_gconv
        # return torch.sigmoid(x_static)*x_static+(1-torch.sigmoid(x_static))*x_gconv
class EmbGCN_SA(nn.Module):
    def __init__(self, dim_in, dim_out, adj,cheb_k, embed_dim):
        super(EmbGCN_SA, self).__init__()
        self.cheb_k = cheb_k
        self.sym_norm_Adj_matrix = torch.from_numpy(sym_norm_Adj(adj)).to(torch.float32).to(torch.device('cuda'))
        self.sym_norm_Adj_matrix=F.softmax(self.sym_norm_Adj_matrix)
        # self.linear=nn.Linear(dim_in, dim_out,bias=True)
        # self.sym_norm_Adj_matrix=F.softmax(torch.Tensor(adj).to(torch.device('cuda')))
        self.SA=Spatial_Attention_layer(adj.shape[0],dim_in,dim_out)
        # self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        # self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        # x=self.SA(x)
        # node_num = node_embeddings.shape[0]
        # supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1) # N N
        # support_set = [torch.eye(node_num).to(supports.device), supports]
        # supports = torch.stack(support_set, dim=0)
        # 静态邻接矩阵
        # x_static = torch.einsum("nm,bmc->bmc",torch.softmax(self.sym_norm_Adj_matrix,dim=-1),x)
        # x_static = self.linear(x_static)
        #
        x_sa = self.SA(x,self.sym_norm_Adj_matrix)
        x_sa=F.relu(x_sa)



        # weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        # bias = torch.matmul(node_embeddings, self.bias_pool)#N, dim_out
        # supports=torch.einsum('knm,mm->knm',supports,self.sym_norm_Adj_matrix)
        #加空注

        # x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        #
        # x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_sa
        # return x_gconv+(torch.tanh(x_gconv)+torch.sigmoid(x_static))
        # return x_gconv+torch.sigmoid(x_static)*x_static
        # return x_gconv+torch.sigmoid(x_static)*x_static+torch.sigmoid(x_gconv)*x_gconv
        # return torch.sigmoid(x_static)*x_static+(1-torch.sigmoid(x_static))*x_gconv


if __name__=="__main__":
    x=torch.randn(64,170,1)
    adj=torch.randn(170,170)
    emb=torch.randn(170,2)
    gcn=EmbGCN(1,1,adj,2,2)
    out=gcn(x,emb)
    print(out.shape)
