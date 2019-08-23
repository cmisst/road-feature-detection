import torch
import torchvision as tv
import numpy as np
import os, warnings
from PIL import Image

class SatelliteDataset(torch.utils.data.dataset.Dataset):
    """ Satellite dataset """

    img_list = []
    root_dir = None
    transform = None
    label_pos = 0

    def __init__(self, img_path, label_pos=-8, transform=None):
        """
        Args:
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.root_dir = img_path
        self.img_list = os.listdir(img_path)
        self.transform = transform
        self.label_pos = label_pos

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.img_list[idx])
        img_as_img = Image.open(img_name)
        img_as_img = img_as_img.convert('RGB')
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)

        # for binary classification
        label = int(self.img_list[idx][self.label_pos], 2)
        filename = np.int64(self.img_list[idx][0:8])
        return (img_as_tensor, label, filename)


if __name__ == "__main__":
    t = tv.transforms.Compose([tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
    dataset = SatelliteDataset('./data/CNN_static_data/Train/', transform=t)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=False)
    for i, (images, labels) in enumerate(dataloader):
        print(images, '\n', labels)
