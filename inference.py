import torch
from llm import Model, encode, decode, device

model = Model()
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.to(device)

prompt = """Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
"""
x = torch.tensor(encode(prompt),dtype=torch.long,device=device).view(1,-1) # [1,len(prompt)]

print(decode(model.generate(x, max_new_tokens=1000)[0].tolist()))