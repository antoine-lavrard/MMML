
import torchaudio
import torch


class NCM:
    """
    Adapted from EASY.
    """
    def __init__(self, n_ways, n_shots):
        self.n_ways = n_ways
        self.nshots = n_shots

    #def __call__(self, train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    def __call__(self, runs):
        bs = runs.shape[0]
        dim = runs.shape[-1]
        
        # targets = torch.arange(self.n_ways, device = runs.device).unsqueeze(1).unsqueeze(0)
        # features = preprocess(train_features, features, elements_train=elements_train)
        # for batch_idx in range(n_runs // batch_few_shot_runs):
        # runs = generate_runs(features, run_classes, run_indices, batch_idx)
        means = torch.mean(runs[:,:,:self.n_shots], dim = 2)
        distances = torch.norm(runs[:,:,self.n_shots:].reshape(bs, self.n_ways, 1, -1, dim) - means.reshape(bs, 1, self.n_ways, 1, dim), dim = 4, p = 2)
        winners = torch.min(distances, dim = 2)[1]
        # winners : bs x n_ways x n_queries
        return winners
    
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, n_class):
        self.n_class = n_class
        self.images = [[] * self.n_class]
        self.target_image = []
        self.is_compiled = False

    def add_feature(self, images, targets):
        with torch.no_grad():
            for image, target in zip(images, targets):
                self.class_to_feature[target].append(image)

    def compile_dataset(self):
        self.is_compiled = True

        for number_class, number_image in enumerate([len(images) for images in self.class_to_feature ]):
            self.target_image += [number_class] * number_image
        
        self.images = torch.concat([
            torch.stack(images, axis=0) for images in self.class_to_feature
        ], axis = 0)

    def __getitem__(self, idx):
        return self.images[idx], self.target_image[idx]
        
    def __len__(self):
        return len(self.images)

