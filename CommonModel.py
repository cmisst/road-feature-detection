import torch
import torchvision
import numpy as np


def load_pt_or_npy(filename, channel_first=True):
    ''' load npy: [N, H, W, C] and permute to [N, C, H, W] '''
    try:
        return torch.load(filename + '.pt')
    except FileNotFoundError:
        data = np.load(filename + '.npy')
        if len(data.shape)==1:
            data = [np.array([data[i]], dtype=np.float16) for i in range(data.shape[0])]
        elif channel_first:
            data = np.swapaxes(data, 1, 3)
            data = [data[i,:,:,:]/256 for i in range(data.shape[0])]
        tensor = torch.stack([torch.Tensor(i) for i in data])
        torch.save(tensor, filename+'.pt')
        return tensor

def main():
    model = torchvision.models.vgg11(pretrained=False, progress=True)
    dataset = torch.utils.data.TensorDataset(load_pt_or_npy('Train-dataset'),
        load_pt_or_npy('Train-labels', channel_first=False))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
    return


if __name__ == "__main__":
    main()
