import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import time

a=np.array(range(0,9999999))
b=np.array([9])

start = time.clock()
c=a-b
print(c.argmin())
elapsed = (time.clock() - start)
print("Time :",elapsed)

a=torch.Tensor(a).cuda()
b=torch.Tensor(b).cuda()

start = time.clock()
c=a-b
print(c.argmin().cpu().numpy())
elapsed = (time.clock() - start)
print("Time :",elapsed)