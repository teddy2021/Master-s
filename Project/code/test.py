import UNet.Unet as Unet
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
	def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
		self.img_labels = pd.read_csv(annotations_file, names=['image_file', 'model_file'])
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, idx):
		img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
		image = read_image(img_path)
		label = self.img_labels.iloc[idx, 1]
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		return image.float(), label



net = Unet.Unet([4,8,16,32,64,128,256], [256,128,64,32,16,8])
data = CustomDataset('UNet/test.csv', 'UNet/')
dataset = DataLoader(data)
a,b = next(iter(dataset))
print(a.shape)
res = net.forward(a)
plt.imshow(res[0].T.detach().numpy())
plt.show()
