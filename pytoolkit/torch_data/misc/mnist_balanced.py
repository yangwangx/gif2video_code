import torch
import torch.utils.data as DD
import torchvision
from collections import defaultdict
import random

class mnist_balanced(DD.Dataset):
    def __init__(self, root='data/mnist/', transform=torchvision.transforms.ToTensor()):
        super(mnist_balanced, self).__init__()
        mnist = torchvision.datasets.MNIST(root=root, train=True, transform=transform, download=True)

        lb2idxs = defaultdict(list)
        for i, v in enumerate(mnist.train_labels.tolist()):
            lb2idxs[v].append(i)

        self.mnist = mnist
        self.lb2idxs = lb2idxs
        self.shuffle()

    def shuffle(self):
        for key in self.lb2idxs:
            random.shuffle(self.lb2idxs[key])

    def __getitem__(self, index):
        idx, cls = divmod(index, 10)
        n = len(self.lb2idxs[cls])
        x1, y1 = self.mnist[self.lb2idxs[cls][idx%n]]
        x2, y2 = self.mnist[random.choice(self.lb2idxs[cls])] # x1, x2 share class
        return x1, y1, x2, y2

    def __len__(self):
        return len(self.mnist)

if __name__ == '__main__':
    trSet = mnist_balanced(root='data/mnist/')
    trSet.shuffle()
    trLD = DD.DataLoader(trSet, batch_size=10, sampler=DD.sampler.SequentialSampler(trSet))
