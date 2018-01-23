import torch.utils.data as data
import PIL
import os

class ImageList(data.Dataset):
    def __init__(self, root, path, transform=None, target_transform=None):
        self.root = root
       
        if os.path.isfile(path):
            with open(path, 'r') as listfile:
                lines = listfile.readlines()

        self.imgs = []
        self.targets = set()

        for l in lines:
            img_path, target = l.replace('\n', '').split(" ")
            self.targets.add(int(target))
            self.imgs.append((img_path, int(target)))

        # Warning! The test data must have the same exact number of labels, use set_targets to avoid problems
        self.set_targets(self.targets)

        self.transform = transform
        self.target_transform = target_transform

    def set_targets(self, targets):
        self.class2idx = { v: i for i, v in enumerate(sorted(targets)) }

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = PIL.Image.open(os.path.join(self.root, path)).convert("RGB")
        target = self.class2idx[target]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)



