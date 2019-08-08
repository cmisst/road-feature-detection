import torch
import torchvision
import numpy as np
import time

# import matplotlib.pyplot as plt


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


def train(trainloader, net, criterion, optimizer, device, epochs=20):
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
            # plt.imshow(images[0].permute(2,1,0))
            # plt.show()
            images = images.to(device)
            labels = labels.to(device)
            # TODO: zero the parameter gradients
            # TODO: forward pass
            # TODO: backward pass
            # TODO: optimize the network
            optimizer.zero_grad()
            scores = torch.sigmoid(net.fc(net.forward(images)))
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
                print(scores.view([1, scores.shape[0]]))

            torch.cuda.empty_cache()
        # print(histogram)
    print('Finished Training')


def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net.fc(net(images))
            outputs = (outputs>0).type(labels.dtype)
            total += labels.shape[0]
            correct += (outputs==labels).sum().item()
        print(correct, total, correct/total)
    pass


def main():
    # Pytorch hardware selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device.type)
    torch.nn.Module.dump_patches = True
    torch.manual_seed(0)
    if device.type=='cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # transform to normalize
    normalize = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])

    # load training dataset
    dataset = torch.utils.data.TensorDataset(load_pt_or_npy('Train-dataset'),
        load_pt_or_npy('Train-labels', channel_first=False))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50)



    # NN configuration
    model = torchvision.models.squeezenet1_0(pretrained=False, progress=True)
    model.fc = torch.nn.Linear(1000, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    print(model)


    train(dataloader, model.to(device), torch.nn.BCELoss().cuda(), optimizer, device)


    # Test
    dataset = torch.utils.data.TensorDataset(load_pt_or_npy('Test-dataset'),
        load_pt_or_npy('Test-labels', channel_first=False))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100)

    test(dataloader, model.to(device), device)



    return


if __name__ == "__main__":
    main()
