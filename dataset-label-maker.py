import os, warnings
from PIL import Image
import numpy as np


labels = []
dataset = []

dimension = (224, 224)
batch = 100
input_path = './data/CNN_static_data/Train/'
output_name = input_path.split('/')[-2]
if os.path.isfile(output_name + '-dataset.npy'):
    warnings.warn(output_name + '-dataset.npy already exists')

for image_name in os.listdir(input_path):
    labels.append(int(image_name[-8], 2))
    image = Image.open(input_path + image_name)
    image = np.asarray(image.resize(dimension).convert('RGB'), dtype=np.uint8)
    dataset.append(image)

augment = batch - len(labels)%batch
for i in range(augment):
    dataset.append(np.zeros(dataset[0].shape, dtype=np.uint8))
    labels.append(0)

np.save(output_name + '-dataset.npy', dataset)
np.save(output_name + '-labels.npy', np.array(labels))
