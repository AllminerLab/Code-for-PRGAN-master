import argparse
import os                              # file route arangeMemt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from DataSet import * #Dataset Module preprocessing data
from MFNNmodule import *
from time import time

np.random.seed(4321)
torch.manual_seed(4321)
torch.cuda.manual_seed(4321)
torch.cuda.manual_seed_all(4321)


# define argparse , control input system parameters
def get_parse_args():
    parser = argparse.ArgumentParser(description="Run PRGAN")
    parser.add_argument('--model', nargs='?', default='PRMF',
                        help='Choose model: PRMF.')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='Use cuda or not.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs.')
    parser.add_argument('--data_path', nargs='?', default='data/lastfm/',
                        help='Data path.')
    parser.add_argument('--valid_record', type=float, default=0.,
                        help='Records greater than the threshold are seen as valid interactions.')
    parser.add_argument('--pre_epochs', type=int, default=3,
                        help='Number of pre-training epochs.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--num_positems', type=int, default=2,
                        help='Number of positive items p in traindata.')
    parser.add_argument('--num_pnitems', type=int, default=13,
                        help='Number of positive and negtive items k in traindata.')
    parser.add_argument('--bs', type=int, default=512,
                        help='Batch size for learning.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for parameters.')
    parser.add_argument('--wd', type=float, default=0.01,
                        help='Weight decay for parameters.')
    parser.add_argument('--p', type=float, default=0.6,
                        help='Dropout rate for Discriminator.')
    parser.add_argument('--c', type=float, default=0.006,
                        help='Weight clipping for Discriminator.')
    return parser.parse_args()

# ---------- data preparation -------

def getdata( args):
    Data = loadDataset( args)
    trainMat = Data.trainMatrix
    num_users = Data.num_users
    num_items = Data.num_items
    trainList = Data.trainList
    testList = Data.testList
    print("the number of users and items is:", num_users, num_items)
    return  trainMat, num_users, num_items, trainList, testList



if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = get_parse_args()
    for k,v in sorted(vars(args).items()):
        print(k,'=',v)


    #start to train
    modelpath = 'modelpara/'
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)

    num_epochs = args.epochs

    # get data set
    trainMat, num_users, num_items, trainList, testList = getdata(args)
    evalItems, user_train_dict = get_eval_negdata(args.data_path, trainList, testList, trainMat)
    
    #start to train

    print('------------------ready to build modle')
    Genmodel = Gen(num_users, num_items, args)
    Memmodel = Gen(num_users, num_items, args)
    Dismodel = Dis(num_users, num_items, args)
    
    Genmodel.cuda()
    Memmodel.cuda()
    Dismodel.cuda()
    
    Genmodel.eval()
    hr, ndcg = testmodel(Genmodel, testList, evalItems, 10)
    Memmodel.eval()
    hr_dis, ndcg_dis = testmodel(Memmodel, testList, evalItems, 10)
    hr_max=max(hr, hr_dis)
    ndcg_max=max(ndcg, ndcg_dis)
    
    for num in range(args.pre_epochs):
        Disloss, Memmodel, Dismodel = PretrainDis(args, Genmodel, Memmodel, Dismodel, trainMat, trainList, user_train_dict)
        print('pretrain epoch:', num ,'; DisLoss:', Disloss)

    for epoch in range(num_epochs):
        epoch_begin = time()
        Genmodel.train()
        Memmodel.train()
        Dismodel.train()

        e_Genloss, Genmodel, e_Disloss, Memmodel, Dismodel = trainGAN(args, Genmodel, Memmodel, Dismodel, trainMat, trainList, user_train_dict)

        if e_Disloss==0:
            break
        Genmodel.eval()
        Memmodel.eval()
        Dismodel.eval()
        hr, ndcg = testmodel(Genmodel, testList, evalItems, 10)
        print('epoch: %d, test on Genmodel, hr:%.4f, ndcg:%.4f' %(epoch, hr, ndcg))
        hr_dis, ndcg_dis = testmodel(Memmodel, testList, evalItems, 10)
        print('epoch: %d, test on Memmodel, hr:%.4f, ndcg:%.4f' %(epoch, hr_dis, ndcg_dis))

        if hr>=hr_max:
            hr_max=hr
            ndcg_max=ndcg
            state = {'Genmodel': Genmodel.state_dict(),'Memmodel': Memmodel.state_dict(), 'Dismodel': Dismodel.state_dict(),'epoch': epoch}
            torch.save(state, modelpath + 'modelpara.pth')
        if hr_dis>=hr_max:
            hr_max=hr_dis
            ndcg_max=ndcg_dis
            state = {'Genmodel': Genmodel.state_dict(),'Memmodel': Memmodel.state_dict(), 'Dismodel': Dismodel.state_dict(),'epoch': epoch}
            torch.save(state, modelpath + 'modelpara.pth')
            
        if epoch%1 == 0 and hr>hr_dis:
            Memmodel.load_state_dict(Genmodel.state_dict(),strict=True)
    
    print('=====================  end training  TOP10  ========================')
    if os.path.exists(modelpath+'modelpara.pth'):
        checkpoint = torch.load(modelpath+'modelpara.pth')
        Genmodel.load_state_dict(checkpoint['Genmodel'],strict=True)
        Memmodel.load_state_dict(checkpoint['Memmodel'],strict=True)
        Dismodel.load_state_dict(checkpoint['Dismodel'],strict=True)
        valid_epoch = checkpoint['epoch']
        print('using model perameter in epoch :',valid_epoch)

    Genmodel.eval()
    Memmodel.eval()
    hr, ndcg = testmodel(Genmodel, testList, evalItems, 10)
    print('test on Genmodel, hr:%.4f, ndcg:%.4f' %(hr, ndcg))
    hr_dis, ndcg_dis = testmodel(Memmodel, testList, evalItems, 10)
    print('test on Memmodel, hr:%.4f, ndcg:%.4f' %(hr_dis, ndcg_dis))