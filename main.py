# This is the inter subscript of whole project

import os

import DataLoader.DataLoader as Loader
import LogSys.LogSys as LogSys
import torch


# 初始化PyTorch框架
LogSys.Print("PyTorch Version: "+torch.__version__)
LogSys.Print("CUDA Version: "+torch.version.cuda)
LogSys.Print("Is CUDA available: " + str(torch.cuda.is_available()))
LogSys.Print("GPU Device Info: " + torch.cuda.get_device_name(0))