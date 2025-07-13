from math import sqrt
import torch

B = 2 # batch size
T = 132 # sequence length i.e. chunk size
d = 120 # embedding dimension
nh = 6  # num heads
assert d % nh == 0 
hd = d // nh # head dimension
n_blocks = 3 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model:
    def __init__():
        pass

    def train():
        pass

    def generate():
        pass 

class MultiHeadAttention(torch.nn.Module):
    """
    X [B,T,d] --> Linear Layer[d,nh*3*hd] --> QKV [B,T,nh*3*hd] Batch -> Token -> Head 1 QKV, Head 2 QKV...
    QKV view and permute --> qkv [B,nh,T,3,hd] Batch -> Head -> Token -> QKV
    Q [B,nh,T,hd]
    K [B,nh,T,hd]
    V [B,nh,T,hd]

    Q [B,nh,T,hd] @ K.T [B,nh,hd,T] --> AM [B,nh,T,T]
    AM [B,nh,T,T] @ V [B,nh,T,hd] --> OUT [B,nh,T,hd]
    OUT permute and view --> OUT [B,T,nh*hd] Batch -> Token -> Concatenated head outputs
    OUT [B,T,nh*hd] @ LL [nh*hd,d] --> self attn out [B,T,d]
    
    """
    def __init__(self):
        super(MultiHeadAttention,self).__init__()
        self.qkv_proj = torch.nn.Linear(d, nh*3*hd, bias=True) # Linear layer for QKV projections
        self.ll = torch.nn.Linear(nh*hd, d, bias=True) # Linear layer for concatenated head output
        self.to(device)

    def forward(self,X):
        QKV = self.qkv_proj(X) # do QKV linear layer transformation
        qkv = QKV.view(B,T,nh,3,hd).permute(0,2,1,3,4).contiguous() # group by token 
        
        Q = qkv[:,:,:,0,:] # all Q matrices 
        K = qkv[:,:,:,1,:] # all K matrices 
        V = qkv[:,:,:,2,:] # all V matrices
        
        AM = (Q @ K.transpose(-2,-1)) / sqrt(d) # attention matrix
        mask = torch.triu(torch.ones_like(AM),diagonal=1).bool()
        masked_AM = AM.masked_fill(mask,float('-inf')) # masking upper triangle for causal self attention
        attn_weights = torch.nn.functional.softmax(masked_AM,dim=-1) # softmax
        out = attn_weights @ V
    
        # make contiguous in memory for view operation later
        out_permute  = out.permute(0,2,1,3).contiguous() # [B,nh,T,hd] --> [B, T, nh, hd]
        if not out_permute.is_contiguous():
            raise Exception('Permuted tensor is not contiguous...')
        out_heads_concat = out_permute.view(B,T,nh*hd) # [B, T, nh, hd] --> [B,T,nh*hd]
        
        self_attention_out = self.ll(out_heads_concat) # [B,T,d]
        return self_attention_out

if __name__ == "__main__":
    mha = MultiHeadAttention()
    X = torch.randn(B,T,d).to(device)
    out = mha(X)
    print(out.shape)
    print(out)