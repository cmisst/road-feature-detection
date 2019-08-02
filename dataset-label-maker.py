import sys, os
from PIL import Image
import numpy as np
import warnings

labels = []
dataset = []

dimension = (224, 224)
input_path = './data/CNN_static_data/Test/'
output_name = input_path.split('/')[-2]
if os.path.isfile(output_name + '-dataset.npy'):
    warnings.warn(output_name + '-dataset.npy already exists')

for image_name in os.listdir(input_path):
    labels.append(int(image_name[-8], 2))
    image = Image.open(input_path + image_name)
    image = np.asarray(image.resize(dimension).convert('RGB'), dtype=np.uint8)
    dataset.append(image)

np.save(output_name + '-dataset.npy', dataset)
np.save(output_name + '-labels.npy', labels)
