"""
This module provides various utility functions for hyperspectral image classification, including data loading, preprocessing, and patch extraction.

Functions:
    zeroPadding_3D(old_matrix, pad_length, pad_depth=0): Pads a 3D matrix with zeros.
    indexToAssignment(index_, Row, Col, pad_length): Converts indices to assignment coordinates.
    assignmentToIndex(assign_0, assign_1, Row, Col): Converts assignment coordinates back to indices.
    selectNeighboringPatch(matrix, pos_row, pos_col, ex_len): Extracts a neighboring patch from a matrix.
    sampling(groundTruth): Samples indices for each class from the ground truth labels.
    load_data_HDF(image_file, label_file): Loads hyperspectral image data and labels from HDF files.
    load_data(image_file, label_file): Loads hyperspectral image data and labels from MAT files.
    getDataAndLabels(trainfn1, trainfn2): Prepares the data and labels for training.
"""

import numpy as np
from sklearn.decomposition import PCA
import random
import pickle
import h5py
#import hdf5storage
from sklearn import preprocessing
import scipy.io as sio

def zeroPadding_3D(old_matrix, pad_length, pad_depth=0):
    """
    Pads a 3D matrix with zeros.
    
    Parameters:
        old_matrix (ndarray): The original matrix.
        pad_length (int): The length of padding on height and width dimensions.
        pad_depth (int): The depth of padding on the depth dimension.
        
    Returns:
        ndarray: The padded matrix.
    """
    new_matrix = np.lib.pad(old_matrix, ((pad_length, pad_length), (pad_length, pad_length), (pad_depth, pad_depth)), 'constant', constant_values=0)
    return new_matrix

def indexToAssignment(index_, Row, Col, pad_length):
    """
    Converts indices to assignment coordinates.
    
    Parameters:
        index_ (list): List of indices.
        Row (int): Number of rows in the original matrix.
        Col (int): Number of columns in the original matrix.
        pad_length (int): Padding length.
        
    Returns:
        dict: A dictionary with index as keys and assignment coordinates as values.
    """
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex(assign_0, assign_1, Row, Col):
    """
    Converts assignment coordinates back to indices.
    
    Parameters:
        assign_0 (int): Row assignment.
        assign_1 (int): Column assignment.
        Row (int): Number of rows in the original matrix.
        Col (int): Number of columns in the original matrix.
        
    Returns:
        int: The corresponding index.
    """
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    """
    Extracts a neighboring patch from a matrix.
    
    Parameters:
        matrix (ndarray): The original matrix.
        pos_row (int): Row position of the center of the patch.
        pos_col (int): Column position of the center of the patch.
        ex_len (int): The half-length of the patch size.
        
    Returns:
        ndarray: The extracted patch.
    """
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def sampling(groundTruth):
    """
    Samples indices for each class from the ground truth labels.
    
    Parameters:
        groundTruth (ndarray): The ground truth labels.
        
    Returns:
        list: A list of sampled indices.
    """
    labels_loc = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices

    whole_indices = []
    for i in range(m):
        whole_indices += labels_loc[i]

    np.random.shuffle(whole_indices)
    return whole_indices

# def load_data_HDF(image_file, label_file):
#     """
#     Loads hyperspectral image data and labels from HDF files.
    
#     Parameters:
#         image_file (str): Path to the image file.
#         label_file (str): Path to the label file.
        
#     Returns:
#         tuple: Scaled data and ground truth labels.
#     """
#     image_data = hdf5storage.loadmat(image_file)
#     label_data = hdf5storage.loadmat(label_file)
#     data_all = image_data['chikusei']  # data_all: ndarray(2517, 2335, 128)
#     label = label_data['GT'][0][0][0]  # label: (2517, 2335)

#     [nRow, nColumn, nBand] = data_all.shape
#     print('chikusei', nRow, nColumn, nBand)
#     gt = label.reshape(np.prod(label.shape[:2]), )
#     del image_data, label_data, label

#     data_all = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104, 204)
#     print(data_all.shape)
#     data_scaler = preprocessing.scale(data_all)
#     data_scaler = data_scaler.reshape(2517, 2335, 128)

#     return data_scaler, gt

def load_data(image_file, label_file):
    """
    Loads hyperspectral image data and labels from MAT files.
    
    Parameters:
        image_file (str): Path to the image file.
        label_file (str): Path to the label file.
        
    Returns:
        tuple: Scaled data and ground truth labels.
    """
    # image_data = sio.loadmat(image_file)
    # label_data = sio.loadmat(label_file)

    f = h5py.File(image_file, 'r')
    # Assuming the data key is the same as the file name without extension
    data_key = list(f.keys())[0]  # Change this if the key is known
    data_all = np.array(f[data_key])
    print(data_all.shape)
    
    f = h5py.File(label_file, 'r')
    # Assuming the label key is the same as the file name without extension
    label_key = list(f.keys())[0]  # Change this if the key is known
    label = np.array(f[label_key])
    print(label_key.shape)
    

    # data_key = image_file.split('/')[-1].split('.')[0]
    # label_key = label_file.split('/')[-1].split('.')[0]
    # data_all = image_data[data_key]
    # label = label_data[label_key]
    gt = label.reshape(np.prod(label.shape[:2]), )

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104, 204)
    print(data.shape)
    data_scaler = preprocessing.scale(data)
    data_scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    return data_scaler, gt

def getDataAndLabels(trainfn1, trainfn2):
    """
    Prepares the data and labels for training.
    
    Parameters:
        trainfn1 (str): Path to the image file.
        trainfn2 (str): Path to the label file.
        
    Returns:
        dict: A dictionary containing the data, labels, and set information.
    """
    # if ('Chikusei' in trainfn1 and 'Chikusei' in trainfn2):
    #     Data_Band_Scaler, gt = load_data_HDF(trainfn1, trainfn2)
    # else:
    #     Data_Band_Scaler, gt = load_data(trainfn1, trainfn2)

    Data_Band_Scaler, gt = load_data(trainfn1, trainfn2)


    del trainfn1, trainfn2
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    # SSRN
    patch_length = 4 # neighbor 9 x 9
    whole_data = Data_Band_Scaler
    padded_data = zeroPadding_3D(whole_data, patch_length)
    del Data_Band_Scaler

    np.random.seed(1334)

    whole_indices = sampling(gt)
    print('the whole indices', len(whole_indices))  # 520

    nSample = len(whole_indices)
    x = np.zeros((nSample, 2 * patch_length + 1, 2 * patch_length + 1, nBand))
    y = gt[whole_indices] - 1  # label 1-19->0-18

    whole_assign = indexToAssignment(whole_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    print('indexToAssignment is ok')
    for i in range(len(whole_assign)):
        x[i] = selectNeighboringPatch(padded_data, whole_assign[i][0], whole_assign[i][1], patch_length)
    print('selectNeighboringPatch is ok')

    print(x.shape)
    del whole_assign, whole_data, padded_data

    imdb = {
        'data': np.zeros([nSample, 2 * patch_length + 1, 2 * patch_length + 1, nBand], dtype=np.float32),
        'Labels': np.zeros([nSample], dtype=np.int64),
        'set': np.zeros([nSample], dtype=np.int64),
    }

    for iSample in range(nSample):
        imdb['data'][iSample, :, :, :] = x[iSample, :, :, :]
        imdb['Labels'][iSample] = y[iSample]
        if iSample % 100 == 0:
            print('iSample', iSample)

    imdb['set'] = np.ones([nSample]).astype(np.int64)
    print('Data is OK.')

    return imdb

# train_data_file = '/home/dell/lm/RN-NEW/datasets/Chikusei/HyperspecVNIR_Chik'
# train_label_file = '/home/dell/lm/RN-NEW/datasets/Chikusei/HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat'

# imdb = getDataAndLabels(train_data_file, train_label_file)#14类

# with open('datasets/Chikusei_imdb_128.pickle', 'wb') as handle:
#     pickle.dump(imdb, handle, protocol=4)

# print('Images preprocessed')