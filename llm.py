from math import sqrt
import torch
import datetime

### Hyperparameters ############################
B = 64 # batch size
T = 256 # sequence length i.e. chunk size
d = 384 # embedding dimension
nh = 6  # num heads
assert d % nh == 0 
hd = d // nh # head dimension
n_blocks = 6
dropout = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_iters = 5000 
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
#################################################

### Data Prep ####################################
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# use character level tokenizer so classification head param count stays small
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - T, (B,)) # generate B random sequence starting points
    x = torch.stack([data[i:i+T] for i in ix])
    y = torch.stack([data[i+1:i+T+1] for i in ix]) # Y is same sequence as X just shifted by 1 for next token prediciton and loss calculation
    x, y = x.to(device), y.to(device)
    return x, y # x and Y are [B,T]
################################################

### Transformer Decoder Code ###################
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.Sequential(*[Block() for _ in range(n_blocks)])
        self.classification_head = torch.nn.Linear(d,vocab_size,bias=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.token_embedding_table = torch.nn.Embedding(vocab_size, d) # token embedding lookup table
        self.position_embedding_table = torch.nn.Embedding(T, d)# pos embedding lookup table

    def forward(self,idx,targets=None):
        tok_emb = self.token_embedding_table(idx) # get embedding for each token in idx [B,T]
        # tok emb [B,T,d]
        pos_emb = self.position_embedding_table(torch.arange(T)) # get embedding for each token position in idx [B,T]
        # pos emb [T,d]
        X = tok_emb + pos_emb # [B,T,d]
        blocks_out = self.blocks(self.dropout(X)) # [B,T,d]
        logits = self.classification_head(blocks_out) # [B,T,vocab_size]

        if targets is None:
            loss = None
        else:
            logits = logits.view(B*T,vocab_size) # view s.t. each row is a token's vocab scores, for every token in every sequence of the batch
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits,targets) # softmax applied in cross_entropy

        return logits,loss

    @torch.no_grad()
    def generate(self,idx,max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            # idx [B,T]
            idx_cond = idx[:, -T:] # get last T tokens to use as context to generate next best token
            # recall that since we train the model on sequence length T, when generating we can only generate the next token given the last T tokens
            logits, loss = self(idx_cond)
            last_token_logits = logits[:,-1,:] # [B,vocab_size] but should always be 1 for generation
            probs = torch.nn.functional.softmax(last_token_logits, dim=-1) # create probability distribution for next token [B,vocab_size] 
            idx_next = torch.multinomial(probs, num_samples=1) # sample from the prob dist to get idx of next token [B, 1] 
            idx = torch.cat((idx, idx_next), dim=1) # concat next token context [B, T+1]
        self.train()
        return idx # return final generated tensor of tokens

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
        super().__init__()
        self.qkv_proj = torch.nn.Linear(d, nh*3*hd, bias=True) # Linear layer for QKV projections
        self.ll = torch.nn.Linear(nh*hd, d, bias=True) # Linear layer for concatenated head output
        self.attn_drop = torch.nn.Dropout(dropout) # weights dropout
        self.proj_drop = torch.nn.Dropout(dropout) # output dropout

    def forward(self,X):
        QKV = self.qkv_proj(X) # do QKV linear layer transformation
        qkv = QKV.view(B,T,nh,3,hd).permute(0,2,1,3,4).contiguous() # group by token 
        
        Q = qkv[:,:,:,0,:] # all Q matrices 
        K = qkv[:,:,:,1,:] # all K matrices 
        V = qkv[:,:,:,2,:] # all V matrices
        
        AM = (Q @ K.transpose(-2,-1)) / sqrt(hd) # attention matrix
        mask = torch.triu(torch.ones_like(AM),diagonal=1).bool()
        masked_AM = AM.masked_fill(mask,float('-inf')) # masking upper triangle for causal self attention
        attn_weights = torch.nn.functional.softmax(masked_AM,dim=-1) # softmax
        attn_weights = self.attn_drop(attn_weights)
        out = attn_weights @ V
    
        # make contiguous in memory for view operation later
        out_permute  = out.permute(0,2,1,3).contiguous() # [B,nh,T,hd] --> [B, T, nh, hd]
        if not out_permute.is_contiguous():
            raise Exception('Permuted tensor is not contiguous...')
        out_heads_concat = out_permute.view(B,T,nh*hd) # [B, T, nh, hd] --> [B,T,nh*hd]
        
        self_attention_out = self.ll(out_heads_concat) # [B,T,d]
        return self.proj_drop(self_attention_out) 
    

class FFNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.ffnn = torch.nn.Sequential(
            torch.nn.Linear(d,4*d),
            torch.nn.GELU(),
            torch.nn.Linear(4*d,d),
            torch.nn.Dropout(dropout)
        )
    
    def forward(self,X):
        logits = self.ffnn(X)
        return logits 
    
class Block(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(d)
        self.ln_2 = torch.nn.LayerNorm(d)
        self.ffnn = FFNN()
        self.mha = MultiHeadAttention()

    def forward(self,X):
        X = X + self.mha(self.ln_1(X))
        X = X + self.ffnn(self.ln_2(X))
        return X

############################################

# TODO: use buffer for masked fill?

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


torch.manual_seed(27)
model = Model()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters") # print total model parameters 

context = torch.zeros((1, 1), dtype=torch.long, device=device) # [B,T] == [1,1] Context for generation

# main training loop
print("starting training...")
for iter in range(max_iters):

    #eval train and val loss every once in a while
    if iter % eval_interval == 0 or iter == max_iters-1:
        loss_estimates = estimate_loss()
        allocated_mb = torch.cuda.memory_allocated() / (1024**2) # allocated memory by PyTorch tensors in mb
        reserved_mb = torch.cuda.memory_reserved() / (1024**2) # memory reserved by PyTorch's caching allocator in mb
        print(f"step {iter}: train loss {loss_estimates['train']:.4f}, val loss {loss_estimates['val']:.4f}")
        print(f"gpu memory allocated: {allocated_mb:.2f} mb, gpu memory reserved: {reserved_mb:.2f} mb")
        print(decode(model.generate(context, max_new_tokens=250)[0].tolist())) # generate from model to see intermediate outputs

    xb,yb = get_batch('train')
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True) # set_to_none=True saves some memory
    loss.backward() # backprop
    optimizer.step() # gradient descent step

print("training done.")
print(f"max gpu memory allocated during training: {torch.cuda.max_memory_allocated() / (1024**2):.2f} mb")
print(f"max gpu memory reserved during training: {torch.cuda.max_memory_reserved() / (1024**2):.2f} mb")

# save model state 
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S") # format to 'YYYY-MM-DD_HH-MM-SS'
torch.save(model.state_dict(),f"model_{formatted_datetime}.pth")

# generate from model
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
open(f'more_{formatted_datetime}.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))