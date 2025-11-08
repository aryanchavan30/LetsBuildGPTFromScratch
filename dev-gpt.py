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