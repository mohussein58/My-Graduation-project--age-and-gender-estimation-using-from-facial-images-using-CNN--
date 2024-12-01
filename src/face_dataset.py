import cv2 as cv
import os
from random import shuffle
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from random import randint
from time import time, strftime
import torch
import shutil


class MemoryDataset(Dataset):
    # Load entries into dataframe in memory - faster than reading each file
    # every time we need to access it, but requires enough RAM to store

    def __init__(self, dir, label_func, transform, processor=None,
                 ds_size=None, print_errors=False, min_size=30,
                 delete_bad_files=False, equalise=False, augment=False):
        all_paths = []
        for root, _, files in os.walk(dir):
            for f in files:
                all_paths.append(os.path.join(root, f))
        shuffle(all_paths)

        if ds_size is None: ds_size = len(all_paths)

        print('Reading' if processor is None else 'Reading and processing',
            f'{ds_size} files from {dir} into memory...')
        self.dataframe = []

        pbar = tqdm(total=ds_size, position=0, leave=False)

        path_idx = 0
        while len(self.dataframe) < ds_size:
            if path_idx >= len(all_paths):
                path_idx = 0
            path = all_paths[path_idx]
            path_idx += 1
            
            filename = os.path.basename(path)

            # Get image class label from filename
            try:
                label = label_func(filename)
            except Exception:
                continue

            if label is None:
                if print_errors: print(f'Skipping {filename}:',
                                    'label function returned None')
                if delete_bad_files: os.remove(path)
                continue

            try:
                image = cv.imread(path)
                if processor:
                    face_images, _ = processor(image)
                    image = face_images[0] # should only have one face in image

                image = Image.fromarray(image) # transform expects PIL image
                image = transform(image)

                entry = {'image': image, 'label': label}
                self.dataframe.append(entry)

                pbar.update(1)
                    
            except Exception as e:
                if print_errors: print(f'Skipping file {filename}: {e}')
                if delete_bad_files: os.remove(path)
                continue

        print(f'\n{len(self.dataframe)} items successfully prepared')
        
        print() # newline
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        entry = self.dataframe[index]
        return entry['image'], entry['label']
    

class EqMemoryDataset(Dataset):
    # EQUALISED VERSION
    
    # Load entries into dataframe in memory - faster than reading each file
    # every time we need to access it, but requires enough RAM to store

    def __init__(self, dir, label_func, transform, processor=None,
                 ds_size=None, print_errors=False, min_size=30,
                 delete_bad_files=False):
        all_paths = []
        for root, _, files in os.walk(dir):
            for f in files:
                all_paths.append(os.path.join(root, f))
        shuffle(all_paths)

        if ds_size is None: ds_size = len(all_paths)

        print('Reading' if processor is None else 'Reading and processing',
            f'{ds_size} files from {dir} into memory...')
        self.dataframe = []

        pbar = tqdm(total=ds_size, position=0, leave=False)

        # Discover classes and init eq requirements
        classes = []
        class_count = {'1': 0}
        for path in all_paths:
            filename = os.path.basename(path)
            try:
                label = label_func(filename)
            except Exception:
                continue
            if label is not None:
                if label not in classes:
                    classes.append(label)
        class_goal = ds_size//len(classes)
        print(f'Equalising - {class_goal} entries per class',
                f'({len(classes)} classes)')

        path_idx = 0

        while min(class_count.values()) < class_goal:
            if path_idx >= len(all_paths):
                path_idx = 0
            path = all_paths[path_idx]
            path_idx += 1
            
            filename = os.path.basename(path)

            # Get image class label from filename
            try:
                label = label_func(filename)
            except Exception:
                continue

            if label is None:
                if print_errors: print(f'Skipping {filename}:',
                                    'label function returned None')
                if delete_bad_files: os.remove(path)
                continue
            elif str(label) in class_count and class_count[str(label)] >= class_goal:
                continue # Skip if we have enough of this class

            try:
                image = cv.imread(path)
                if processor:
                    face_images, _ = processor(image)
                    image = face_images[0] # [0] is most prevalent face in image (usually only one)

                image = Image.fromarray(image) # transform expects PIL image
                image = transform(image)

                entry = {'image': image, 'label': label}
                self.dataframe.append(entry)

                if str(label) not in class_count:
                    class_count[str(label)] = 0
                else:
                    class_count[str(label)] += 1

                pbar.update(1)
                    
            except Exception as e:
                if print_errors: print(f'Skipping file {filename}: {e}')
                if delete_bad_files: os.remove(path)
                continue

        print(f'\n{len(self.dataframe)} items successfully prepared')
        print(f'Equalised datset to {class_goal} images per class')
        print(f'{class_count}')
        
        print() # newline
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        entry = self.dataframe[index]
        return entry['image'], entry['label']
    

class EqStorageDataset(Dataset):
    # Saves processed images to disk, loads each time they are requested
    # Slightly slower than MemoryDataset, but allows for larger datasets

    def __init__(self, dir, label_func, transform, processor=None,
                 ds_size=None, print_errors=False, min_size=30,
                 delete_bad_files=False, equalise=False, augment=False):

        og_paths = []
        for root, _, files in os.walk(dir):
            for f in files:
                og_paths.append(os.path.join(root, f))
        shuffle(og_paths)

        if ds_size is None: ds_size = len(og_paths)

        print('Reading' if processor is None else 'Reading and processing',
            f'{ds_size} files from {dir} ...')
        self.dataframe = []

        self.processor = processor
        self.transform = transform

        pbar = tqdm(total=ds_size, position=0, leave=False)

        # Discover classes and init eq requirements
        classes = []
        for path in og_paths:
            filename = os.path.basename(path)
            try:
                label = label_func(filename)
            except Exception:
                continue
            if label is not None:
                if label not in classes:
                    classes.append(label)
        class_goal = ds_size//len(classes)
        class_count = {}
        aug_count = {}
        for c in classes:
            class_count[str(c)] = 0
            aug_count[str(c)] = 0
        del classes
        if equalise:
            print(f'Equalising - {class_goal} entries per class')

        path_idx = 0
        cycles = 0

        while len(self.dataframe) < ds_size:
            if path_idx >= len(og_paths):
                path_idx = 0
                cycles += 1
            path = og_paths[path_idx]
            path_idx += 1
            
            filename = os.path.basename(path)

            # Get image class label from filename
            try:
                label = label_func(filename)
            except Exception:
                continue

            if label is None:
                if print_errors: print(f'Skipping {filename}:',
                                    'label function returned None')
                if delete_bad_files: os.remove(path)
                continue

            elif equalise and class_count[str(label)] > class_goal:
                continue # Skip if we have enough of this class

            try:
                entry = {'filename': path, 'label': label}
                self.dataframe.append(entry)

                class_count[str(label)] += 1
                pbar.update(1)
                    
            except Exception as e:
                if print_errors: print(f'Skipping file {filename}: {e}')
                if delete_bad_files: os.remove(path)
                continue

        print(f'\n{len(self.dataframe)} items successfully prepared')
        if print_errors:
            if equalise:
                print(f'Equalised datset to {class_goal} images per class')
            print(f'Final class counts: {class_count}')
        
        print() # newline
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        entry = self.dataframe[index]
        # Load the image tensor saved earlier
        path = entry['filename']
        
        try:
            image = cv.imread(path)
            if self.processor:
                face_images, _ = self.processor(image)
                image = face_images[0] # [0] is most prevalent face in image (usually only one)

            image = Image.fromarray(image) # transform expects PIL image
            image = self.transform(image)
                
        except Exception as e:
            return self.__getitem__(index+1) # try again
        
        return image, entry['label']


# old original function
class SlowDataset(Dataset):
    # Saves processed images to disk, loads each time they are requested
    # Slightly slower than MemoryDataset, but allows for larger datasets
    def __init__(self, dir, label_func, transform, processor=None,
                 print_errors=False, min_size=30, ds_size=None,
                 delete_bad_files=False, equalise=False, classes=None,
                 augment=False):
        self.dir = dir
        self.label_func = label_func
        self.processor = processor
        self.transform = transform
        self.print_errors = print_errors
        self.min_size = min_size
        self.delete_bad_files = delete_bad_files
        self.equalise = equalise
        self.classes = classes
        self.ds_size = ds_size
        self.augment = augment
        self.count = 0

        self.all_paths = []
        for root, _, files in os.walk(dir):
            for f in files:
                self.all_paths.append(os.path.join(root, f))
        shuffle(self.all_paths)

        if ds_size and ds_size < len(self.all_paths):
            self.all_paths = self.all_paths[:ds_size]

        print('Found and shuffled',
            f'{len(self.all_paths)} files from {dir}...')
        # Discover classes
        classes = []
        for path in self.all_paths:
            filename = os.path.basename(path)
            label = label_func(filename)
            if label is not None:
                classes.append(label)
        # init eq requirements
        class_goal = ds_size//len(classes)
        eq_requirements = []
        for c in classes:
            requirement = {'class': str(c), 'count': 0, 'aug_count': 0}
            eq_requirements.append(requirement)
        del classes
        if equalise:
            print(f'Equalising - {class_goal} entries per class')
        
        print() # newline

    def __len__(self):
        return self.ds_size

    def __getitem__(self, index):
        path = self.all_paths[index % len(self.all_paths)]

        try:
            filename = os.path.basename(path)

            # Get image class label from filename
            label = self.label_func(filename)

            if label is None:
                if self.print_errors: print(f'Skipping {filename}:',
                                    'label function returned None')
                if self.delete_bad_files: os.remove(path)
                return self.__getitem__(index+1)
            
            elif self.equalise and self.eq_requirements[str(label)]['count'] > self.class_goal:
                return self.__getitem__(index+1) # skip if we have enough of this class

            image = cv.imread(path)
            if self.processor:
                face_images, coords = self.processor(image)
                image = face_images[0]

            image = Image.fromarray(image) # transform expects PIL image
            if self.augment and index > len(self.all_paths): # Have used all data in dataset
                    # Randomly apply various augmentations to bolster dataset
                image = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAdjustSharpness(sharpness_factor=randint(0,10)),
                    transforms.GaussianBlur(randint(0, 10)*2+1),
                    transforms.RandomRotation(randint(0,10), fill=255),
                    transforms.RandomPerspective(distortion_scale=0.2, fill=255),
                    transforms.RandomGrayscale(p=0.3),
                    self.transform,
                ])(image)
            else:
                image = self.transform(image)
            
        except Exception as e:
            if self.print_errors: print(f'Skipping file {filename}: {e}')
            if self.delete_bad_files: os.remove(path)
            return self.__getitem__(index+1)
    
        return image, label