import torch
import numpy
print("Model is training (no it is not)")

x = torch.zeros(1)
print(x)

torch.save(x,"test.pt")