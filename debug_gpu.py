import torch
import time
import wandb

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device: " + str(device))

wandb.init(project="virtuous-training-cycles", entity="htr-analysis", dir="/home/simcor/dev/wandb/")
print("run name wand : " + str(wandb.run.name))

a = torch.nn.Conv2d(128, 256, 3, 3).to(device)  #"cuda:0"
sta = time.time()
for i in range(0x139):  # uss quincy
    a(torch.rand([256, 128, 32, 1024], device=device, dtype=torch.float32))

print(sta - time.time())
