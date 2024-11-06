from torch.utils.data import Dataset
import torch


class ObjectsDataset(Dataset):
    """
    Creates dataloader for object-centric data.

    Args:
        X: numpy array of observations
        Z: numpy array of ground-truth latents
        transform: torchvision transformation for data

    Returns:
        inferred batch of observations and ground-truth latents
    """

    def __init__(self, X, Z, transform):
        self.obs = X
        self.factors = Z
        self.transform = transform

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        x = self.obs[idx]
        # import pdb; pdb.set_trace()
        #turn x into a tensor where each image is transformed using self.transform
        num_obj = x.shape[0]
        H = x.shape[1]
        W = x.shape[2]
        C = x.shape[3]
        x_tensor = torch.zeros(num_obj, C, H, W)
        if self.transform != None:
            for i in range(x.shape[0]):
                x_tensor[i] = self.transform(x[i])

        factors = self.factors[idx]
        return x_tensor, factors
