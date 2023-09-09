'''
Processing datasets.
@author: Wen Jing
'''
import os
import numpy as np
import scipy.sparse as sp
import csv

np.random.seed(4321)

class loadDataset(object):
    '''
    Loading the data file
        trainMatrix: load rating records as sparse matrix for class Data
        trainList: load rating records as list to speed up user's feature retrieval
        testRatings: load leave-one-out rating test for class Evaluate
    '''

    def __init__(self, args):
        '''
        Constructor
        '''
        self.path = args.data_path
        self.valid_rating = args.valid_record
        self.trainList = self.load_train_rating_file_as_list(self.path + "traindata.csv")
        self.trainMatrix = self.load_rating_file_as_matrix(self.trainList)
        self.testList = self.load_test_rating_file_as_list(self.path + "testdata.csv")
        self.num_users, self.num_items = self.trainMatrix.shape


    def load_train_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            lines = csv.reader(f, delimiter='\t')
            if lines != None and lines != "":
                for row in lines:
                    user = int(row[0])
                    item = int(row[1])
                    rating = float(row[2])
                    if rating > self.valid_rating :
                        ratingList.append([user, item, 1.])
        f.close()
        print('filename:',filename)
        num_data = len(ratingList)
        print('number of training data is :', num_data) 
        return ratingList

    def load_test_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            lines = csv.reader(f, delimiter='\t')
            if lines != None and lines != "":
                for row in lines:
                    user = int(row[0])
                    item = int(row[1])
                    rating = float(row[2])
                    ratingList.append([user, item, 1.])
        f.close()
        print('filename:',filename)
        return ratingList


    def load_rating_file_as_matrix(self, ratingList):
        # Get number of users and items
        max_userID, max_itemID = 0, 0
        for row in ratingList:
            u = int(row[0])
            i = int(row[1])
            max_userID = max(max_userID, u)
            max_itemID = max(max_itemID, i)

        # Construct matrix
        mat = sp.dok_matrix(( max_userID+1, max_itemID+1), dtype=np.float32)
        for row in ratingList:
            user,item,rating = int(row[0]),int(row[1]),float(row[2])
            if rating > 0.:
                mat[user, item] =1.
        return mat


def get_eval_negdata( path, trainList, testList, trainMat):
    num = len(testList)
    num_users, num_items = trainMat.shape
    assert num == int(num_users)
    user_train_dict = get_user_train_dict(trainList)

    if os.path.exists(path + 'testnegatives.csv'):
        print('There are evaluating items.')
        evalItem_list = load_negative_file(path + 'testnegatives.csv')
    return evalItem_list, user_train_dict


def get_user_train_dict(trainList):
    user_item_set = dict()
    for data in trainList:
        u, i = int(data[0]), int(data[1])
        if u in user_item_set:
            user_item_set[u].add(i)
        else:
            user_item_set[u] = set([i])
    return user_item_set


def load_negative_file(filename):
    negativeList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1:]:
                negatives.append(int(x))
            assert len(negatives)==99
            negativeList.append(negatives)
            line = f.readline()
    print('length:', len(negativeList), filename)
    return negativeList


def sample_train_data_list( trainMatrix, trainList, user_train_list, args):
    x_userID = []
    x_itemset = []
    y_r = []
    num_users, num_items = trainMatrix.shape
    for data in trainList:
        user = data[0]
        itemset = []
        r = []
        posdata = list(user_train_list[user])
        len_pos = len(posdata)
        num_pos = min(len_pos,args.num_positems)
        # find rated item
        itemset.append(data[1])
        r.append(1.)
        for i in range(1,num_pos):
            j = np.random.randint(len_pos)
            while posdata[j] in itemset:
                j = np.random.randint(len_pos)
            itemset.append(posdata[j])
            r.append(1.)
        
        # find unrated item
        num_neg = int(args.num_pnitems-num_pos)
        for i in range(num_neg):
            j = np.random.randint(num_items)
            while ((user,j) in trainMatrix.keys()) or (j in itemset):
                j = np.random.randint(num_items)
            itemset.append(j)
            r.append(0.)
        x_userID.append([user])
        x_itemset.append(list(itemset))
        y_r.append(r)
        
    return x_userID,x_itemset,y_r