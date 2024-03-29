import torch
import torchvision as tv
import numpy as np
import time
from SatelliteDataset import SatelliteDataset

# import matplotlib.pyplot as plt




def train(trainloader, net, criterion, optimizer, device, epochs):
    for epoch in range(epochs):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        # try:
        #     histogram = np.zeros(np.shape(trainloader.dataset.dataset.classes))
        # except AttributeError:
        #     histogram = np.zeros(np.shape(np.unique(trainloader.dataset.dataset.train_labels)))
        for i, (images, labels, filename) in enumerate(trainloader):
            # histogram += np.histogram(labels.numpy(), bins=np.shape(histogram)[0])[0]
            # print(torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            # plt.imshow(images[0].permute(2,1,0))
            # plt.show()
            images = images.to(device)
            labels = labels.type(torch.float)
            labels = labels.to(device)
            # TODO: zero the parameter gradients
            # TODO: forward pass
            # TODO: backward pass
            # TODO: optimize the network
            optimizer.zero_grad()
            scores = torch.sigmoid(net.fc(net.forward(images)))
            loss = criterion(scores.reshape(labels.shape), labels)
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
    confusion_matrix = torch.zeros(2,2)
    with torch.no_grad():
        for data in testloader:
            images, labels, filenames = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net.fc(net(images))
            outputs = (outputs>0).type(labels.dtype)
            outputs = outputs.reshape(labels.shape)
            for t,p in zip(labels, outputs):
                confusion_matrix[t.long(), p.long()] += 1
        print(confusion_matrix.numpy())
    pass

class CommonModel():
    net = None
    device = None
    transform = None
    optimizer = None

    def __init__(self):
        # Pytorch hardware selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device.type)
        torch.nn.Module.dump_patches = True
        torch.manual_seed(0)
        if self.device.type=='cuda':
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # NN configuration
        self.net = tv.models.squeezenet1_1(pretrained=True, progress=True)
        self.net.fc = torch.nn.Linear(1000, 1)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        print(self.net)

    def set_data_transform(self, t=None):
        if t is None:
            t = []
        t = [tv.transforms.Resize((224,224))] + t
        t.append(tv.transforms.ToTensor())
        self.transform = tv.transforms.Compose(t)

    def train(self, label_pos, batch, epochs=10, load_pretrained=False, path=None):
        if path is None:
            path = './data/CNN_static_data/Train/'

        # load training dataset
        dataset = SatelliteDataset(path, label_pos=label_pos, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch)

        train(dataloader, self.net.to(self.device), torch.nn.BCELoss().cuda(), 
            self.optimizer, self.device, epochs)

    def test(self, label_pos, batch, path=None):
        if path is None:
            path = './data/CNN_static_data/Test/'

        # load test dataset
        dataset = SatelliteDataset(path, label_pos=label_pos, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch)

        test(dataloader, self.net.to(self.device), self.device)

    def predict(self, batch, path=None):
        # load unlabeled dataset
        dataset = SatelliteDataset(path, label_pos=2, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch)
        
        n = np.array([]) # filenames
        p = np.array([]) # predict
        c = np.array([]) # confidence
        with torch.no_grad():
            start=time.time()
            for data in dataloader:
                images, labels, filenames = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net.fc(self.net(images))
                outputs = outputs.reshape(labels.shape)
                c = np.append(c, torch.sigmoid(outputs/10).cpu().numpy())
                outputs = (outputs>0).type(labels.dtype)
                p = np.append(p, outputs.cpu().numpy())
                n = np.append(n, filenames.cpu().numpy())
                end = time.time()
                print(n.shape, 'elapsed time: {}'.format(end-start))
                start = time.time()
        c[p==0] = 1 - c[p==0]
        return np.int64(np.stack((n,p, c*100)).T)


def main():
    prefix='./data/'
    transforms=[[],
        [tv.transforms.RandomVerticalFlip(p=1)],
        [tv.transforms.RandomHorizontalFlip(p=1)],
        [tv.transforms.RandomHorizontalFlip(p=1), tv.transforms.RandomVerticalFlip(p=1)]
    ]
    # m = CommonModel()
    # for t in transforms*20:
    #     m.set_data_transform(t=t)
    #     m.train(label_pos=-5, batch=100, epochs=1, path=prefix+'Median_Satellite_View/train/')
    # torch.save(m.net, 'median.pth')
    # m.net = torch.load('median.pth')
    # m.test(label_pos=-5, batch=200, path=prefix+'Median_Satellite_View/test/')
    # m.test(label_pos=-5, batch=200, path=prefix+'Median_Satellite_View/Remainder1/')

    m = CommonModel()
    m.set_data_transform(t=[])
    for t in transforms*100:
        m.set_data_transform(t=t)
        m.train(label_pos=-8, batch=300, epochs=1, path=prefix+'Crosswalk_Satellite_View/train/')
        print(t)
    torch.save(m.net, 'crosswalk.pth')
    m.net = torch.load('crosswalk.pth')
    m.test(label_pos=-8, batch=200, path=prefix+'Crosswalk_Satellite_View/test/')
    m.test(label_pos=-8, batch=200, path=prefix+'Crosswalk_Satellite_View/Remainder0/')
    # np.savetxt('predict_median.csv', 
    #     m.predict(batch=2000, path=prefix+'Unlabeled_Satellite_IMG/'),
    #     fmt='%d', delimiter=',')
    return






if __name__ == "__main__":
    main()
