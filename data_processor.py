import numpy as np 
import utils_doc as utils 
import math 
import torch 


def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]

    [Row, Column] = np.nonzero(G)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    train = {}
    test = {}
    da_train = {}
    m = int(np.max(G))
    nlabeled = shot_num_per_class
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))
    print('the number of test_indices:', len(test_indices))
    print('the number of train_indices after data argumentation:', len(da_train_indices))
    print('labeled sample indices:', train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices
    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                             Column[RandPerm[iSample]] - HalfWidth:Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]]
        if iSample < nTrain:
            imdb['set'][iSample] = 1

    imdb['data'] = imdb['data'].transpose((3, 2, 0, 1))
    imdb['Labels'] = imdb['Labels'].reshape(-1)
    imdb['set'] = imdb['set'].reshape(-1)

    imdb_da = {}
    imdb_da['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain + nTest], dtype=np.float32)
    imdb_da['Labels'] = np.zeros([da_nTrain + nTest], dtype=np.int64)
    imdb_da['set'] = np.zeros([da_nTrain + nTest], dtype=np.int64)

    da_RandPerm = da_train_indices + test_indices
    da_RandPerm = np.array(da_RandPerm)

    for iSample in range(da_nTrain + nTest):
        imdb_da['data'][:, :, :, iSample] = data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
                                                Column[da_RandPerm[iSample]] - HalfWidth:Column[da_RandPerm[iSample]] + HalfWidth + 1, :]
        imdb_da['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]]
        if iSample < da_nTrain:
            imdb_da['set'][iSample] = 1

    imdb_da['data'] = imdb_da['data'].transpose((3, 2, 0, 1))
    imdb_da['Labels'] = imdb_da['Labels'].reshape(-1)
    imdb_da['set'] = imdb_da['set'].reshape(-1)

    return imdb, imdb_da, HalfWidth



def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    """
    Prepares the target dataset for few-shot learning and domain adaptation in hyperspectral image classification.

    Parameters:
    Data_Band_Scaler (numpy.ndarray): The scaled hyperspectral image data.
    GroundTruth (numpy.ndarray): The ground truth labels for the hyperspectral image.
    class_num (int): The number of classes in the target dataset.
    shot_num_per_class (int): The number of labeled samples per class for few-shot learning.

    Returns:
    train_loader (DataLoader): DataLoader for training data.
    test_loader (DataLoader): DataLoader for testing data.
    target_da_metatrain_data (dict): Meta-training data for domain adaptation.
    target_loader (DataLoader): DataLoader for target domain data.
    G: Unspecified return value from get_train_test_loader.
    RandPerm: Unspecified return value from get_train_test_loader.
    Row: Unspecified return value from get_train_test_loader.
    Column: Unspecified return value from get_train_test_loader.
    nTrain: Unspecified return value from get_train_test_loader.
    """
    train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain = get_train_test_loader(
        Data_Band_Scaler=Data_Band_Scaler, 
        GroundTruth=GroundTruth, 
        class_num=class_num,
        shot_num_per_class=shot_num_per_class
    )
    
    # Fetch and display training data and labels
    train_datas, train_labels = train_loader.__iter__().next()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)  # size of train datas: torch.Size([45, 103, 9, 9])

    # Display dataset information
    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # Data augmentation for target data
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # (9,9,100,1800)->(1800,100,9,9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    print('target data augmentation label:', target_da_labels)

    # Meta-training data preparation for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # Prepare target domain batch samples for domain adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain
