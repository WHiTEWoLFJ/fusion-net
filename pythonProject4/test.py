import time

import numpy
import torch
from numpy import double
from torchvision import transforms
from models import networks, data_loader
from models.lossfunction import PsnrLoss, SSIM
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from model.zh_model import SingleModel
from data.data_loader import *
from util.visualizer import Visualizer
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from numpy.core.defchararray import zfill
from model.network import CFMSAN
import util.util as util
def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream)


if __name__ == '__main__':
    opt = TrainOptions().parse()
    config = get_config(opt.config)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    model = SingleModel(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    skip = True if opt.skip > 0 else False
    model.d_net = CFMSAN()
    model.h_net = CFMSAN()
    print("---is not train----")
    which_epoch = opt.which_epoch
    print("---model is loaded---")

    model.load_network(model.d_net, 'G_V')
    model.load_network(model.h_net, 'G_H')
    print('---------- Networks initialized -------------')
    model.d_net.eval()
    model.h_net.eval()
    total_psnr = 0
    total_ssim = 0
    total_num = 0
    for i in range(249, 359):
        imgA_path = "" + str(i) + ".png"
        imgB_path = "" + str(i) + ".png"

        img_fused = "" + str(i) + ".png"

        try:
            imgA = cv2.imread(imgA_path)
            imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
            imgB = cv2.imread(imgB_path)
            imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)
            total_num += 1
        except:
            continue
        imgA = double(imgA) / 255
        imgB = double(imgB) / 255

        imgA = torch.from_numpy(imgA)
        imgB = torch.from_numpy(imgB)

        imgA = imgA.unsqueeze(0)
        imgB = imgB.unsqueeze(0)
        imgA = imgA.permute(0, 3, 2, 1).float()
        imgB = imgB.permute(0, 3, 2, 1).float()
        a1=model.d_net.forward(imgA)
        a2=model.h_net.forward(imgB)
        output1=imgA*a1+imgB*a2

        output = output1.cpu()
        outputnumpy=util.latent2im(output.data)
        outputnumpy=cv2.flip(cv2.transpose(outputnumpy),1)
        outputnumpy = cv2.resize(outputnumpy, [341,512])
        outputnumpy=cv2.flip(outputnumpy,1)
        outputimage = Image.fromarray(numpy.uint8(outputnumpy))
        outputimage.save(img_fused)
        print("ok")
