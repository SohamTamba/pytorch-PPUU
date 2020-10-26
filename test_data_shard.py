import torch
z = torch.load("traffic-data/state-action-cost/data_i80_v0/trajectories-0500-0515/all_data.pth")
print("data shard = ", z['ego_car'][:10])
