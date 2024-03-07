import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST

# Download training dataset
datasets = MNIST(root='data/', train=True, download=True)
print(datasets[0])

ret = transforms.ToTensor()(datasets[0][0])
print(ret.size())
# img = datasets[0][0]
# img.show()
