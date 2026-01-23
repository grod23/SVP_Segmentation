from torch.utils.data import Dataset
import torch
import cv2
import matplotlib.pyplot as plt
import sys

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
        plt.title('Image Before Preprocessing')
        plt.axis('off')
        plt.imshow(image.permute(1, 2, 0).detach().cpu().numpy() / 255.0)
        plt.subplot(1, 2, 2)
        plt.title('Mask Before Preprocessing')
        plt.axis('off')
        plt.imshow(mask.permute(1, 2, 0).detach().cpu().numpy() / 255.0)
        plt.show()

        # Extract green plane from image
        green_plane = image[1, :, :]
        print(f'Green Shape: {green_plane.shape}')
        print("Green min/max:", green_plane.min(), green_plane.max())
        print(f'Green Type: {type(green_plane)}')

        plt.figure(figsize=(10, 10))
        plt.title('Green Plane')
        plt.axis('off')
        plt.imshow(green_plane.detach().cpu().numpy())
        plt.show()

        # CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        green_np = green_plane.cpu().numpy().round().astype('uint8')
        enhanced_image = clahe.apply(green_np)
        print(f'Enhanced Shape: {enhanced_image.shape}')
        print('Enhanced min/max:', enhanced_image.min(), enhanced_image.max())
        print(f'Enhanced Type: {type(enhanced_image)}')

        plt.figure(figsize=(10, 10))
        plt.title('CLAHE Image')
        plt.axis('off')
        plt.imshow(enhanced_image)
        plt.show()

        sys.exit()

        return image, mask

        # https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2024.1470941/full

        # PREPROCESSING:
        # 1.) High Pass Filtering - Given RGB fundus image, separate binary image by the green plane (green plane shows vessels
        # strongest).
        # 2.) CLAHE
        # 3.) Min max normalization or STD Standardization