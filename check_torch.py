import urllib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torchvision
import json

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import coremltools as ct

# Load the model (deeplabv3)
model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True).eval()

# Load a sample image (cat_dog.jpg)
input_image = Image.open("1.jpg")


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():
    output = model(input_batch)['out'][0]
torch_predictions = output.argmax(0)
print(torch_predictions.shape)

class WrappedDeeplabv3Resnet101(nn.Module):

    def __init__(self):
        super(WrappedDeeplabv3Resnet101, self).__init__()
        self.model = torch.hub.load(
            'pytorch/vision:v0.6.0',
            'deeplabv3_resnet101',
            pretrained=True
        ).eval()

    def forward(self, x):
        res = self.model(x)
        # Extract the tensor we want from the output dictionary
        x = res["out"]
        print("THE X SHAPE", x.shape)
        return x
traceable_model = WrappedDeeplabv3Resnet101().eval()
trace = torch.jit.trace(traceable_model, input_batch)

mlmodel = ct.convert(
    trace,
    inputs=[ct.TensorType(name="input", shape=input_batch.shape)],
)