# Few-Shot Hyperspectral Image Classification
"""
This script is designed for few-shot hyperspectral image classification using PyTorch. It loads and preprocesses datasets, sets up hyperparameters, and prepares data loaders for training and testing. The script utilizes utilities from `utils` and `models` modules, which are assumed to be defined elsewhere.
"""

## Imports

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import pickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import time
import utils_doc as utils
import models_doc as models
from utils_doc import euclidean_metric
from const import * 
import math

# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape) # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]

    [Row, Column] = np.nonzero(G)  # (10249,) (10249,)
    # print(Row)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {} # Data Augmentation
    m = int(np.max(G))  # 9
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS
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

    print('the number of train_indices:', len(train_indices))  # 520
    print('the number of test_indices:', len(test_indices))  # 9729
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    print('labeled sample indices:',train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train ,G,RandPerm,Row, Column,nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)  # 9 classes and 5 labeled samples per class
    train_datas, train_labels = train_loader.__iter__().next()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape) # size of train datas: torch.Size([45, 103, 9, 9])

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # (9,9,100, 1800)->(1800, 100, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain



'''

Directory Initialization
Directories for checkpoints and classification maps are created if they don't exist.

'''

utils.same_seeds(0)

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')

_init_()


'''

Load Source Domain Dataset
The source domain dataset is loaded from a pickle file.

'''
# Load preprocessed data
# source_data_path = os.path.join('datasets', 'Chikusei_imdb_128.pickle')

# with open(source_data_path, 'rb') as handle:
#     source_imdb = pickle.load(handle)

# print(source_imdb.keys())
# print(source_imdb['Labels'])


## Process Source Domain Dataset
## The source domain dataset is processed to prepare it for training.

# data_train = source_imdb['data']  # Extract the training data from the source IMDB 
# labels_train = source_imdb['Labels']  # Extract the corresponding labels for the training data 

# print(f'Shape of the training data: {data_train.shape}')  # Print the shape of the training data (77592, 9, 9, 128)
# print(f'Shape of the training labels: {labels_train.shape}')  # Print the shape of the training labels (77592)

## Get all unique classes in the training labels and sort them
# keys_all_train = sorted(list(set(labels_train)))  
# print(keys_all_train)  # Print the sorted list of unique classes [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

# label_encoder_train = {}  # Initialize a dictionary to map class labels to indices

# # Populate the label encoder with class indices
# for i in range(len(keys_all_train)):
#     label_encoder_train[keys_all_train[i]] = i  # Map each class label to a unique index
# print(label_encoder_train)  # Print the label encoder dictionary

# train_set = {}  # Initialize a dictionary to store the training set

# # Organize data paths into the training set based on their corresponding labels
# for class_, path in zip(labels_train, data_train):
#     if label_encoder_train[class_] not in train_set:
#         train_set[label_encoder_train[class_]] = []  # Create a new list for this class if it doesn't exist
#     train_set[label_encoder_train[class_]].append(path)  # Append the samples to the corresponding class list
# print(train_set.keys())  # Print the keys (class indices) of the training set

# data = train_set  # Assign the training set to the variable 'data'
# del train_set  # Delete the 'train_set' variable to free up memory
# del keys_all_train  # Delete the keys_all_train variable to free up memory
# del label_encoder_train  # Delete the label_encoder_train variable to free up memory

# # Print the number of classes in the source domain datasets
# print("Num classes for source domain datasets: " + str(len(data)))  
# print(data.keys())  # Print the keys (class indices) of the 'data'
# data = utils.sanity_check(data)  # Perform a sanity check on the dataset
# # Print the number of classes that are larger than 200
# print("Num classes of the number of class larger than 200: " + str(len(data)))

# # Transpose each image in the dataset to change the channel order
# for class_ in data:
#     for i in range(len(data[class_])):
#         image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # Transpose image from (H, W, C) to (C, H, W)
#         data[class_][i] = image_transpose  # Update the dataset with the transposed image

# metatrain_data = data  # Assign the processed data to 'metatrain_data'
# print(len(metatrain_data.keys()), metatrain_data.keys())  # Print the number of classes and their keys
# del data  # Delete the 'data' variable to free up memory

# # Print the shape of the original source data
# print(source_imdb['data'].shape)  
# source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))  # Transpose the source data to (H, W, C, D)
# print(source_imdb['data'].shape)  # Print the shape of the transposed source data
# print(source_imdb['Labels'])  # Print the labels of the source IMDB

# # Create a source dataset using a utility function
# source_dataset = utils.matcifar(source_imdb, 
#                                 train=True, 
#                                 d=3, 
#                                 medicinal=0)  

# # Create a DataLoader for the source dataset with specified batch size and shuffling
# source_loader = torch.utils.data.DataLoader(source_dataset, 
#                                             batch_size=128,
#                                             shuffle=True,
#                                             num_workers=0)

# del source_dataset, source_imdb  # Delete the source_dataset and source_imdb variables to free up memory


'''

Load and Process Target Domain Dataset
The target domain dataset is loaded and processed to prepare it for training and testing.

'''

# test_data = 'datasets/paviaU/paviaU.mat'
# test_label = 'datasets/paviaU/paviaU_gt.mat'

# Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

crossEntropy = nn.CrossEntropyLoss().to(device)
domain_criterion = nn.BCEWithLogitsLoss().to(device)

# run 10 times
nDataSet = 1
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None

seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]
for iDataSet in range(nDataSet):

    # load target domain data for training and testing
    np.random.seed(seeds[iDataSet])

    (train_loader, 
     test_loader, 
     target_da_metatrain_data, 
     target_loader,
     G,
     RandPerm,
     Row, 
     Column,
     nTrain) = get_target_dataset(Data_Band_Scaler=Data_Band_Scaler, 
                                  GroundTruth=GroundTruth,
                                  class_num=TEST_CLASS_NUM, 
                                  shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    

    # model
    feature_encoder = models.Network(FEATURE_DIM, 
                                     SRC_INPUT_DIMENSION, 
                                     TAR_INPUT_DIMENSION, 
                                     N_DIMENSION, 
                                     CLASS_NUM)
    
    domain_classifier = models.DomainClassifier()

    random_layer = models.RandomLayer([FEATURE_DIM, CLASS_NUM], 1024) 

    feature_encoder.apply(utils.weights_init)
    domain_classifier.apply(utils.weights_init)

    feature_encoder.to(device)
    domain_classifier.to(device)
    random_layer.to(device)  # Random layer

    feature_encoder.train()
    domain_classifier.train()


    # optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    domain_classifier_optim = torch.optim.Adam(domain_classifier.parameters(), lr=LEARNING_RATE)

    

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))

    train_start = time.time()
    
    for episode in range(10000):  # EPISODE = 90000

        # get domain adaptation data from  source domain and target domain
        try:
            source_data, source_label = source_iter.next()
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = source_iter.next()

        try:
            target_data, target_label = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = target_iter.next()

        # source domain few-shot + domain adaptation
        if episode % 2 == 0:
            '''Few-shot classification for source domain data set'''
            # generate task for few shot classification
            task = utils.Task(metatrain_data, CLASS_NUM=5, SHOT_NUM_PER_CLASS=1, QUERY_NUM_PER_CLASS=15)  # 5， 1，15

            # prepare dataloaders
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # extract data samples
            supports, support_labels = support_dataloader.__iter__().next()  # (5, 100, 9, 9)
            querys, query_labels = query_dataloader.__iter__().next()  # (75,100,9,9)

            # encoding feature -> Encoding feature tensor, classfier output tensor
            support_features, support_outputs = feature_encoder(supports.to(device))  # torch.Size([409, 32, 7, 3, 3])
            query_features, query_outputs = feature_encoder(querys.to(device))  # torch.Size([409, 32, 7, 3, 3])
            target_features, target_outputs = feature_encoder(target_data.to(device), domain='target')  # torch.Size([409, 32, 7, 3, 3])

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_features

            # fsl_loss
            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, query_labels.to(device))

            '''domain adaptation'''
            # calculate domain adaptation loss
            features = torch.cat([support_features, query_features, target_features], dim=0)
            outputs = torch.cat((support_outputs, query_outputs, target_outputs), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)

            # set label: source 1; target 0
            domain_label = torch.zeros([supports.shape[0] + querys.shape[0] + target_data.shape[0], 1]).to(device)
            domain_label[:supports.shape[0] + querys.shape[0]] = 1  # torch.Size([225=9*20+9*4, 100, 9, 9])

            randomlayer_out = random_layer.forward([features, softmax_output])  # torch.Size([225, 1024=32*7*3*3])

            domain_logits = domain_classifier(randomlayer_out, episode)
            domain_loss = domain_criterion(domain_logits, domain_label)

            # total_loss = fsl_loss + domain_loss
            loss = f_loss + domain_loss  # 0.01

            # Update parameters
            feature_encoder.zero_grad()
            domain_classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            domain_classifier_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]




        # target domain few-shot + domain adaptation--------------------------------------------------
        else:
            '''Few-shot classification for target domain data set'''
            # get few-shot classification samples
            task = utils.Task(target_da_metatrain_data, 
                              TEST_CLASS_NUM, 
                              SHOT_NUM_PER_CLASS, 
                              QUERY_NUM_PER_CLASS)  # 5， 1，15
            
            support_dataloader = utils.get_HBKC_data_loader(task, 
                                                            num_per_class=SHOT_NUM_PER_CLASS, 
                                                            split="train", 
                                                            shuffle=False)
            
            query_dataloader = utils.get_HBKC_data_loader(task, 
                                                          num_per_class=QUERY_NUM_PER_CLASS, 
                                                          split="test", 
                                                          shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().next()  # (5, 100, 9, 9)
            querys, query_labels = query_dataloader.__iter__().next()  # (75,100,9,9)

            # calculate features
            support_features, support_outputs = feature_encoder(supports.to(device),  domain='target')  # torch.Size([409, 32, 7, 3, 3])
            query_features, query_outputs = feature_encoder(querys.to(device), domain='target')  # torch.Size([409, 32, 7, 3, 3])
            source_features, source_outputs = feature_encoder(source_data.to(device))  # torch.Size([409, 32, 7, 3, 3])

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_features

            # fsl_loss
            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, query_labels.to(device))

            '''domain adaptation'''
            features = torch.cat([support_features, query_features, source_features], dim=0)
            outputs = torch.cat((support_outputs, query_outputs, source_outputs), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)

            domain_label = torch.zeros([supports.shape[0] + querys.shape[0] + source_features.shape[0], 1]).to(device)
            domain_label[supports.shape[0] + querys.shape[0]:] = 1

            randomlayer_out = random_layer.forward([features, softmax_output])  # torch.Size([225, 1024=32*7*3*3])

            domain_logits = domain_classifier(randomlayer_out, episode)  # , label_logits
            domain_loss = domain_criterion(domain_logits, domain_label)

            # total_loss = fsl_loss + domain_loss
            loss = f_loss + domain_loss  # 0.01 0.5=78;0.25=80;0.01=80

            # Update parameters
            feature_encoder.zero_grad()
            domain_classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            domain_classifier_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]

        if (episode + 1) % 100 == 0:  # display
            train_loss.append(loss.item())
            print('episode {:>3d}:  domain loss: {:6.4f}, fsl loss: {:6.4f}, acc {:6.4f}, loss: {:6.4f}'.format(episode + 1, \
                                                                                                                domain_loss.item(),
                                                                                                                f_loss.item(),
                                                                                                                total_hit / total_num,
                                                                                                                loss.item()))

        if (episode + 1) % 1000 == 0 or episode == 0:

            # testing..........
            print("Testing ...")
            # Setting evaluation mode
            train_end = time.time()
            feature_encoder.eval()

            # Initializations
            total_rewards = 0
            counter = 0 # count the total number of test sample
            accuracies = [] # record accuracy for each batch
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)

            # Extract feature for training data
            train_datas, train_labels = train_loader.__iter__().next()
            train_features, _ = feature_encoder(Variable(train_datas).to(device), domain='target')  # (45, 160)
            
            # Feature normalization
            max_value = train_features.max()  # 89.67885
            min_value = train_features.min()  # -57.92479
            print(f'Train Feature max value: {max_value.item()}')
            print(f'Train Feature min value: {min_value.item()}')

            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            ''' Train KNN classifier '''

            # fit the classifier
            KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)  # .cpu().detach().numpy()

            # Test KNN classifier

            # iterate through the test batches
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                # Extract and normalize test batches
                test_features, _ = feature_encoder(Variable(test_datas).to(device), domain='target')  # (100, 160)
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                # predict labels
                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format( total_rewards, len(test_loader.dataset),
                100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode
            feature_encoder.train()
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(),str( "checkpoints/DFSL_feature_encoder_" + "UP_" +str(iDataSet) +"iter_" + str(TEST_LSAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                print("save networks for episode:",episode+1)
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)

                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            print(f'best episode:[{best_episdoe + 1}], best accuracy={last_accuracy}')

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G,best_RandPerm,best_Row, best_Column,best_nTrain = G, RandPerm, Row, Column, nTrain

    print(f'iter:{iDataSet} best episode:[{best_episdoe + 1}], best accuracy={last_accuracy}')
    print('***********************************************************************************')

AA = np.mean(A, 1)

AAMean = np.mean(AA,0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)


print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("accuracy for each class: ")

for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


best_iDataset = 0
for i in range(len(acc)):
    print(f'{i}:{acc[i]}')
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print(f'best acc all={acc[best_iDataset]}')

#################classification map################################

# for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
#     best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[i] + 1

# hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))

# for i in range(best_G.shape[0]):
#     for j in range(best_G.shape[1]):
#         if best_G[i][j] == 0:
#             hsi_pic[i, j, :] = [0, 0, 0]
#         if best_G[i][j] == 1:
#             hsi_pic[i, j, :] = [0, 0, 1]
#         if best_G[i][j] == 2:
#             hsi_pic[i, j, :] = [0, 1, 0]
#         if best_G[i][j] == 3:
#             hsi_pic[i, j, :] = [0, 1, 1]
#         if best_G[i][j] == 4:
#             hsi_pic[i, j, :] = [1, 0, 0]
#         if best_G[i][j] == 5:
#             hsi_pic[i, j, :] = [1, 0, 1]
#         if best_G[i][j] == 6:
#             hsi_pic[i, j, :] = [1, 1, 0]
#         if best_G[i][j] == 7:
#             hsi_pic[i, j, :] = [0.5, 0.5, 1]
#         if best_G[i][j] == 8:
#             hsi_pic[i, j, :] = [0.65, 0.35, 1]
#         if best_G[i][j] == 9:
#             hsi_pic[i, j, :] = [0.75, 0.5, 0.75]
# utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "classificationMap/UP_{}shot.png".format(TEST_LSAMPLE_NUM_PER_CLASS))


