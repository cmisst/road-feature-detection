import torch
import torchvision
import numpy as np
import time


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


def train(trainloader, net, criterion, optimizer, device, epochs=5):
    for epoch in range(epochs):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        # try:
        #     histogram = np.zeros(np.shape(trainloader.dataset.dataset.classes))
        # except AttributeError:
        #     histogram = np.zeros(np.shape(np.unique(trainloader.dataset.dataset.train_labels)))
        for i, (images, labels) in enumerate(trainloader):
            # histogram += np.histogram(labels.numpy(), bins=np.shape(histogram)[0])[0]
            # print(torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            images = images.to(device)
            labels = labels.to(device)
            # TODO: zero the parameter gradients
            # TODO: forward pass
            # TODO: backward pass
            # TODO: optimize the network
            optimizer.zero_grad()
            scores = net.fc(net.forward(images))
            loss = criterion(scores, labels.view(labels.shape[0]))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()

            if i % 5 == 0:    # print every 2000 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch , i , running_loss, end-start))
                start = time.time()
                running_loss = 0.0

            torch.cuda.empty_cache()
        # print(histogram)
    print('Finished Training')



def main():
    # Pytorch hardware selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device.type)
    torch.nn.Module.dump_patches = True
    torch.manual_seed(0)
    if device.type=='cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # load training dataset
    dataset = torch.utils.data.TensorDataset(load_pt_or_npy('Train-dataset'),
        load_pt_or_npy('Train-labels', channel_first=False).long())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # NN configuration
    model = torchvision.models.vgg11(pretrained=False, progress=True)
    model.fc = torch.nn.Linear(1000, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(dataloader, model.to(device), torch.nn.CrossEntropyLoss().cuda(), optimizer, device)


    return


if __name__ == "__main__":
    main()
