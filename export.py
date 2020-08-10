import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import torch.nn as nn

class ModelForExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = (x/255.0)*2.0 - 1.0
        return (self.model(x) + 1.0) / 2.0 * 255

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads =
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    model_for_export = ModelForExport(model.netG.module)
    model.netG.module.eval()
    model.netG.module.to('cpu'
                         )
    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    output = model_for_export(dummy_input.detach())

    input_names = ["input"]
    output_names = ["output"]

    model_for_export = ModelForExport(model.netG.module)

    torch.onnx.export(model_for_export, dummy_input,
                      "./mobilenet.onnx", verbose=False,
                      input_names=input_names, output_names=output_names)

