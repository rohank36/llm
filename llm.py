import math
import torch

class Model:
    B = 2 # batch size
    T = 128 # sequence length i.e. chunk size
    d = 128 # embedding dimension
    hd = 64 # head dimension
    nh = 6  # num heads
    n_blocks = 3 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__():
        pass

    def train():
        pass

    def generate():
        pass 



class MultiHead:
    """
    X [B,T,d] --> Linear Layer[d,nh*3*hd] --> QKV [B,T,nh*3*hd]
    reformat QKV --> qkv [B,nh,T,3,hd] 
    Q [B,nh,T,hd]
    K [B,nh,T,hd]
    V [B,nh,T,hd]

    Q [B,nh,T,hd] @ K.T [B,nh,hd,T] --> AM [B,nh,T,T]
    AM [B,nh,T,T] @ V [B,nh,T,hd] --> OUT [B,nh,T,hd]
    
    """
    def __init__(self):
        pass
    
class Head:
    """
    X [B,T,d] @ Qw [d,pd] = Q [B,T,pd]
    X [B,T,d] @ Kw [d,pd] = K [B,T,pd]
    X [B,T,d] @ Vw [d,d] = V [B,T,d]

    Q [B,T,pd] @ K.T [B,pd,T] = Attn_matrix [B,T,T]
    Attn_matrix [B,T,T] @ V [B,T,d] = Head_output [B,T,d]



    X [B,T,d] @ Qw [B,n_head,T*pd]
    """

    def __init__(self,d=Model.d, hd=Model.hd):
        self.d = d
        self.Qw = torch.randn(d,hd,requires_grad=True)
        self.Kw = torch.randn(d,hd,requires_grad=True)
        self.Vw = torch.randn(d,hd,requires_grad=True)

        self.Qw = self.Qw.to(Model.device)
        self.Kw = self.Kw.to(Model.device)
        self.Vw = self.Vw.to(Model.device)


    def single_head_masked_self_attention(self,Q,K,V,d):
        attn_scores = (Q @ K.transpose(-2,-1)) / math.sqrt(d)
        mask = torch.triu(torch.ones_like(attn_scores),diagonal=1).bool()
        masked_attn_scores = attn_scores.masked_fill(mask,float('-inf'))
        attn_weights = torch.nn.functional.softmax(masked_attn_scores,dim=-1)
        head_output = attn_weights @ V
        return head_output
    
    def forward(self,X):
        Q = X @ self.Qw
        K = X @ self.Kw
        V = X @ self.Vw
        return self.single_head_masked_self_attention(Q,K,V,self.d)
    

class MultiHead:
    pass

class Block: 
    pass 