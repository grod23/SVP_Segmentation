from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

class FundusDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        print('Data Type')
        print(type(self.data[index]))
        batch = self.data[index]
        if self.transform:
            batch = self.transform(batch)
        image = batch['Image']
        mask = batch['Mask']
        print(f'Image Shape: {image.shape}')
        print(f'Image Type: {image.dtype}')
        print("Image min/max:", image.min(), image.max())
        print(f'Mask Shape: {mask.shape}')
        print(f'Mask Type: {mask.dtype}')
        print("Mask min/max:", mask.min(), mask.max())
        print('\n')

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.title('Image')
        plt.axis('off')
        plt.imshow(image.permute(1, 2, 0).detach().cpu().numpy())
        plt.subplot(1, 2, 2)
        plt.title('Mask')
        plt.axis('off')
        plt.imshow(mask.permute(1, 2, 0).detach().cpu().numpy())
        plt.show()

        return image, mask

        # https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2024.1470941/full

        # PREPROCESSING:
        # 1.) High Pass Filtering - Given RGB fundus image, separate binary image by the green plane (green plane shows vessels
        # strongest).