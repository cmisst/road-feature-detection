import torch
import torchvision
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
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels, filenames = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net.fc(net(images))
            outputs = (outputs>0).type(labels.dtype)
            outputs = outputs.reshape(labels.shape)
            total += labels.shape[0]
            correct += (outputs==labels).sum().item()
        print(correct, total, correct/total)
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
        self.net = torchvision.models.squeezenet1_1(pretrained=False, progress=True)
        self.net.fc = torch.nn.Linear(1000, 1)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        print(self.net)

    def set_data_transform(self, t=None):
        if t is None:
            self.transform = torchvision.transforms.Compose(
                [torchvision.transforms.Resize((224,224)),
                torchvision.transforms.ToTensor()])
        else:
            self.transform = t

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
        with torch.no_grad():
            start=time.time()
            for data in dataloader:
                images, labels, filenames = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net.fc(self.net(images))
                outputs = (outputs>0).type(labels.dtype)
                outputs = outputs.reshape(labels.shape)
                n = np.append(n, filenames.cpu().numpy())
                p = np.append(p, outputs.cpu().numpy())
                end = time.time()
                print(n.shape, 'elapsed time: {}'.format(end-start))
                start = time.time()
        return np.int64(np.stack((n,p)).T)


def main():
    m = CommonModel()
    m.set_data_transform()
    prefix='/scratch/engin_root/engin/yugtmath/'
    m.train(label_pos=-5, batch=50, epochs=10, path=prefix+'Median_Satellite_View/train/')
    m.test(label_pos=-5, batch=200, path=prefix+'Median_Satellite_View/test/')
    m.test(label_pos=-5, batch=200, path=prefix+'Median_Satellite_View/Remainder1/')

    m.train(label_pos=-8, batch=50, epochs=10, path=prefix+'Crosswalk_Satellite_View/train/')
    m.test(label_pos=-8, batch=200, path=prefix+'Crosswalk_Satellite_View/test/')
    m.test(label_pos=-8, batch=200, path=prefix+'Crosswalk_Satellite_View/Remainder0/')
    # np.savetxt('predict_median.csv', 
    #     m.predict(batch=1000, path='/scratch/engin_root/engin/yugtmath/Unlabeled_Satellite_IMG/'),
    #     fmt='%d', delimiter=',')
    return






if __name__ == "__main__":
    main()
