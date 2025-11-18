with open(r'Solar_Industries_India_Comprehensive_Analysis.txt','r',encoding='utf-8') as f:
    text = f.read()

print("Lenght of Data :- " , len(text))

print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("My name is Aryan"))
print(decode(encode("My name is Aryan")))

import torch
data = torch.tensor(encode(text), dtype=torch.long)

print(data.shape , data.dtype)
print(data[:1000])


n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8  # COntext window
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f'when input is {context} the target is  {target}')

torch.manual_seed(1473)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    return x , y

xb , yb = get_batch("train")
print("inputs:")
print(xb.shape)
print(xb)
print("target:")
print(yb.shape)
print(yb)

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b , :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target is: {target}")

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1473)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B, T, C)

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)

        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is the (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the prediction
            logits ,loss = self(idx)
            # foucs only on last time step
            logits = logits[:,-1,:] # become (B,C)
            # apply softmax to get prob 
            probs = F.softmax(logits,dim=1) #(B,C)
            #sample from the distribution
            idx_next = torch.multinomial(probs , num_samples = 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx,idx_next),dim=1)
            return idx

m = BigramLanguageModel(vocab_size)
logits , loss = m(xb,yb)
print(logits.shape)
print(loss)