import torch

nn_model = torch.load("/home/dehui2/test/PES-Learn/1_Tutorials/1_water_pes_api/model3_data/model.pt")
out = nn_model(torch.tensor([3, 4, 5], dtype=torch.long))
print("Hello world")