import torch
from torch import nn 
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import scipy as sp
import scipy.stats
import random
import scipy.io as sio
from sklearn import preprocessing
import matplotlib.pyplot as plt


def same_seeds(seed):
    """
    Set the random seed for reproducibility.

    Parameters:
    seed (int): The seed to set for all random number generators.

    Description:
    This function sets the seed for various random number generators to ensure that 
    experiments are reproducible. It covers seeds for PyTorch, NumPy, and Python's 
    random module. It also configures PyTorch's cuDNN backend to ensure deterministic 
    behavior.
    """
    torch.manual_seed(seed)  # Set the seed for PyTorch
    if torch.cuda.is_available():  # If a GPU is available
        torch.cuda.manual_seed(seed)  # Set the seed for the current GPU
        torch.cuda.manual_seed_all(seed)  # Set the seed for all GPUs (if using multi-GPU)

    np.random.seed(seed)  # Set the seed for NumPy
    random.seed(seed)  # Set the seed for Python's random module

    torch.backends.cudnn.benchmark = False  # Disable cuDNN benchmark for deterministic results
    torch.backends.cudnn.deterministic = True  # Ensure deterministic cuDNN behavior

def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate the mean and confidence interval for a dataset.

    Parameters:
    data (list or array): The data for which the mean and confidence interval are calculated.
    confidence (float): The confidence level for the interval (default is 0.95).

    Returns:
    tuple: A tuple containing the mean (m) and the margin of error (h).

    Description:
    This function computes the mean and the confidence interval for the given data 
    using the specified confidence level. It uses the standard error of the mean and 
    the t-distribution to calculate the margin of error.
    """
    a = 1.0 * np.array(data)  # Convert the data to a NumPy array and ensure float type
    n = len(a)  # Get the number of data points
    m, se = np.mean(a), scipy.stats.sem(a)  # Calculate the mean and standard error
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)  # Calculate the margin of error
    return m, h  # Return the mean and the margin of error

from operator import truediv

def AA_andEachClassAccuracy(confusion_matrix):
    """
    Calculate the average accuracy and each class accuracy from a confusion matrix.

    Parameters:
    confusion_matrix (ndarray): The confusion matrix from which to calculate accuracies.

    Returns:
    tuple: A tuple containing each class accuracy (each_acc) and the average accuracy (average_acc).

    Description:
    This function computes the accuracy for each class and the average accuracy 
    from the given confusion matrix. It handles division by zero by using `np.nan_to_num`.
    """
    counter = confusion_matrix.shape[0]  # Get the number of classes
    list_diag = np.diag(confusion_matrix)  # Extract the diagonal elements (true positives)
    list_raw_sum = np.sum(confusion_matrix, axis=1)  # Sum each row (total actual instances per class)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))  # Calculate per-class accuracy
    average_acc = np.mean(each_acc)  # Calculate the average accuracy
    return each_acc, average_acc  # Return per-class and average accuracy


import torch.utils.data as data

class matcifar(data.Dataset):
    """
    `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        imdb (dict): Dictionary containing the dataset.
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        d (int): Dimensionality indicator (2 or 3).
        medicinal (int): Indicator for special dataset handling.

    Description:
    This class handles the CIFAR-10 dataset, allowing for training and testing splits.
    The data can be transformed into 2D or 3D format depending on the 'd' parameter.
    """

    def __init__(self, imdb, train, d, medicinal):
        self.train = train  # Indicates whether the dataset is for training or testing
        self.imdb = imdb  # The dataset dictionary
        self.d = d  # Dimensionality indicator
        self.x1 = np.argwhere(self.imdb['set'] == 1).flatten()  # Indices for the training set
        self.x2 = np.argwhere(self.imdb['set'] == 3).flatten()  # Indices for the test set

        if medicinal == 1:
            self.train_data = self.imdb['data'][self.x1, :, :, :]  # Training data
            self.train_labels = self.imdb['Labels'][self.x1]  # Training labels
            self.test_data = self.imdb['data'][self.x2, :, :, :]  # Test data
            self.test_labels = self.imdb['Labels'][self.x2]  # Test labels
        else:
            self.train_data = self.imdb['data'][:, :, :, self.x1]  # Training data for non-medicinal case
            self.train_labels = self.imdb['Labels'][self.x1]  # Training labels for non-medicinal case
            self.test_data = self.imdb['data'][:, :, :, self.x2]  # Test data for non-medicinal case
            self.test_labels = self.imdb['Labels'][self.x2]  # Test labels for non-medicinal case
            
            # Transpose data based on dimensionality indicator 'd'
            if self.d == 3:
                self.train_data = self.train_data.transpose((3, 2, 0, 1))  # Shape: (17, 17, 200, 10249)
                self.test_data = self.test_data.transpose((3, 2, 0, 1))
            else:
                self.train_data = self.train_data.transpose((3, 0, 2, 1))
                self.test_data = self.test_data.transpose((3, 0, 2, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the index of the target class.

        Description:
        This method retrieves the image and target (label) at the specified index.
        It handles both training and testing data based on the 'train' attribute.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]  # Get training image and label
        else:
            img, target = self.test_data[index], self.test_labels[index]  # Get test image and label

        return img, target

    def __len__(self):
        """
        Returns:
            int: The length of the dataset (number of samples).

        Description:
        This method returns the number of samples in the dataset.
        It checks whether the dataset is for training or testing and returns the appropriate length.
        """
        if self.train:
            return len(self.train_data)  # Length of training data
        else:
            return len(self.test_data)  # Length of test data


def sanity_check(all_set):
    """
    Perform a sanity check on the dataset to ensure each class has at least 200 samples.

    Args:
        all_set (dict): Dictionary where keys are class labels and values are lists of samples.

    Returns:
        dict: Filtered dictionary with classes that have at least 200 samples, each containing exactly 200 samples.
    """
    nclass = 0
    nsamples = 0
    all_good = {}
    for class_ in all_set:
        if len(all_set[class_]) >= 200:
            all_good[class_] = all_set[class_][:200]
            nclass += 1
            nsamples += len(all_good[class_])
    print('the number of classes:', nclass)
    print('the number of samples:', nsamples)
    return all_good

def flip(data):
    """
    Create a larger matrix with the original data centered and padded with zeros.

    Args:
        data (np.array): Original data array.

    Returns:
        np.array: Expanded data array with zero padding.
    """
    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data

def load_data(image_file, label_file):
    """
    Load and preprocess image and label data from .mat files.

    Args:
        image_file (str): Path to the image .mat file.
        label_file (str): Path to the label .mat file.

    Returns:
        tuple: Preprocessed image data and ground truth labels.
    """
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    data_key = data_key[0].lower() + data_key[1:]
    
    data_all = image_data[data_key]
    GroundTruth = label_data[label_key]

    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))
    data_scaler = preprocessing.scale(data)
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    return Data_Band_Scaler, GroundTruth

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    """
    Apply radiation noise to the data.

    Args:
        data (np.array): Original data array.
        alpha_range (tuple): Range for scaling factor alpha.
        beta (float): Scaling factor for the noise.

    Returns:
        np.array: Data with added radiation noise.
    """
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

def flip_augmentation(data):
    """
    Perform random horizontal and vertical flips on the data for augmentation.

    Args:
        data (np.array): Original data array.

    Returns:
        np.array: Augmented data array.
    """
    horizontal = np.random.random() > 0.5
    vertical = np.random.random() > 0.5
    if horizontal:
        data = np.fliplr(data)
    if vertical:
        data = np.flipud(data)
    return data

class Task(object):
    """
    Class representing a few-shot learning task.

    Args:
        data (dict): Dictionary where keys are class labels and values are lists of samples.
        num_classes (int): Number of classes in the task.
        shot_num (int): Number of support samples per class.
        query_num (int): Number of query samples per class.
    """
    def __init__(self, data, num_classes, shot_num, query_num):
        self.data = data
        self.num_classes = num_classes
        self.support_num = shot_num
        self.query_num = query_num

        class_folders = sorted(list(data))
        class_list = random.sample(class_folders, self.num_classes)
        labels = np.array(range(len(class_list)))
        labels = dict(zip(class_list, labels))

        samples = dict()
        self.support_datas = []
        self.query_datas = []
        self.support_labels = []
        self.query_labels = []

        for c in class_list:
            temp = self.data[c]
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.support_datas += samples[c][:shot_num]
            self.query_datas += samples[c][shot_num:shot_num + query_num]
            self.support_labels += [labels[c] for i in range(shot_num)]
            self.query_labels += [labels[c] for i in range(query_num)]

class FewShotDataset(Dataset):
    """
    Few-shot dataset class.

    Args:
        task (Task): Task object containing the data.
        split (str): 'train' for support set, 'test' for query set.
    """
    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.image_datas = self.task.support_datas if self.split == 'train' else self.task.query_datas
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class HBKC_dataset(FewShotDataset):
    """
    HBKC dataset class for few-shot learning.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        """
        Retrieve the image and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is the sample data and label is the corresponding label.
        """
        image = self.image_datas[idx]
        label = self.labels[idx]
        return image, label

class ClassBalancedSampler(Sampler):
    """
    Class-balanced sampler for few-shot learning.

    Args:
        num_per_class (int): Number of samples per class.
        num_cl (int): Number of classes.
        num_inst (int): Total number of instances.
        shuffle (bool): Whether to shuffle the samples.

    Description:
    Samples 'num_inst' examples each from 'num_cl' pool of examples of size 'num_per_class'.
    """
    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]
        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

def get_HBKC_data_loader(task, num_per_class=1, split='train', shuffle=False):
    """
    Get a data loader for the HBKC dataset.

    Args:
        task (Task): Current task.
        num_per_class (int): Number of samples per class.
        split (str): 'train' for support set, 'test' for query set.
        shuffle (bool): Whether to shuffle the samples.

    Returns:
        DataLoader: Data loader for the HBKC dataset.
    """
    dataset = HBKC_dataset(task, split=split)
    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num, shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)
    return loader

def classification_map(map, groundTruth, dpi, savePath):
    """
    Generate and save a classification map.

    Args:
        map (np.array): Classification map data.
        groundTruth (np.array): Ground truth labels.
        dpi (int): Dots per inch for the saved image.
        savePath (str): Path to save the classification map.

    Returns:
        int: 0 indicating success.
    """
    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1] * 2.0 / dpi, groundTruth.shape[0] * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(savePath, dpi=dpi)
    return 0

def euclidean_metric(a, b):
    """
    Computes the Euclidean distance between two sets of vectors.

    Parameters:
    a (torch.Tensor): The first set of vectors with shape (n, d).
    b (torch.Tensor): The second set of vectors with shape (m, d).

    Returns:
    logits (torch.Tensor): The negative squared Euclidean distance between vectors a and b with shape (n, m).
    """
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


def weights_init(m):
    """
    Initializes the weights of the model using different initialization strategies
    based on the layer type.

    Parameters:
    m (nn.Module): A module in the neural network.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())
