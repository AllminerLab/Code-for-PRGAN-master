import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import heapq
import math

np.random.seed(4321)
torch.manual_seed(4321)
torch.cuda.manual_seed(4321)
torch.cuda.manual_seed_all(4321)
torch.backends.cudnn.deterministic = True

#---------- model definition -------
class Dataset(data.TensorDataset):
    def __init__(self, trainMat, trainList, user_train_dict, args):
        super(Dataset,self).__init__()
        self.trainList = trainList
        self.trainMat = trainMat
        self.user_train_dict = user_train_dict
        self.num_users, self.num_items = trainMat.shape
        self.num_positems = args.num_positems
        self.num_pnitems = args.num_pnitems


    def __getitem__(self, index):
        data = self.trainList[index]
        user = data[0]
        itemset = []
        r = []
        posdata = list(self.user_train_dict[user])
        len_pos = len(posdata)
        num_pos = min(len_pos, self.num_positems)
		
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
        num_neg = int(self.num_pnitems-num_pos)
        for i in range(num_neg):
            j = np.random.randint(self.num_items)
            while ((user,j) in self.trainMat.keys()) or (j in itemset):
                j = np.random.randint(self.num_items)
            itemset.append(j)
            r.append(0.)
        
        return torch.tensor(user).long(), torch.tensor(itemset).long(), torch.tensor(r)


    def __len__(self):
        return len(self.trainList)



class Gen(nn.Module):
  def __init__(self, num_users, num_items, args):
    super(Gen,self).__init__()  #继承
    self.embed_size = args.embed_size
    self.num_users = num_users
    self.num_items = num_items
    
    #model components
    self.userembed = torch.nn.Embedding(num_users, self.embed_size)
    nn.init.normal_(self.userembed.weight, mean=0., std=0.01)
    self.itemembed = torch.nn.Embedding(num_items, self.embed_size)
    nn.init.normal_(self.itemembed.weight, mean=0., std=0.01)

  def forward(self, input_userID, input_itemID):
      
      self.Uembed = self.userembed(input_userID)
      self.Iembed = self.itemembed(input_itemID)
      
      #matrix factorization
      y_gen = torch.sum(torch.mul(self.Uembed,self.Iembed),dim=-1)
      return y_gen


class Dis(nn.Module):
  def __init__(self, num_users, num_items, args):
    super(Dis,self).__init__()  #继承
    self.embed_size = args.embed_size
    self.num_pnitems = args.num_pnitems
    self.num_users = num_users
    self.num_items = num_items
    self.p = args.p
    
    #model components
    self.convCol = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2,1)) #out channel*num_pnitems
    self.convRow = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,2)) #out channel*2*(num_pnitems-1)
    self.convSqu = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2,2)) #out channel*(num_pnitems-1)
    self.BNCol = nn.BatchNorm2d(3)
    self.BNRow = nn.BatchNorm2d(3)
    self.BNSqu = nn.BatchNorm2d(3)
    self.ydense1 = nn.Linear(in_features=int(3*(4*self.num_pnitems-3)), out_features=int(3*self.num_pnitems))
    self.ydense2 = nn.Linear(in_features=int(3*self.num_pnitems), out_features=1)
    self.drop = nn.Dropout(self.p)
    
  def forward(self, input_mem ,input_data):

      input_mem1 = torch.unsqueeze(input_mem, dim=1)
      input_data1 = torch.unsqueeze(input_data, dim=1)
      
      input_x = torch.cat((input_mem1,input_data1), dim=1)
      input_x = torch.unsqueeze(input_x,dim=1)
      
      x1 = self.convCol(input_x)
      x1 = self.BNCol(x1)
      x1 = torch.flatten(x1,start_dim=1)
      
      x2 = self.convRow(input_x)
      x2 = self.BNRow(x2)
      x2 = torch.flatten(x2,start_dim=1)
      
      x3 = self.convSqu(input_x)
      x3 = self.BNSqu(x3)
      x3 = torch.flatten(x3,start_dim=1)
      
      x = torch.cat((x1,x2),dim=1)
      x = torch.cat((x,x3),dim=1)
      x = self.ydense1(x)
      x = F.relu(x)
      x = self.drop(x)
      y_pred = self.ydense2(x)
      return y_pred


def PretrainDis(args, Genmodel, Memmodel, Dismodel, trainMat, trainList, user_train_dict):
    dataset = Dataset(trainMat, trainList, user_train_dict, args)  # number of tarining data record
    data_loder = data.DataLoader(dataset=dataset, batch_size=args.bs, shuffle=True, num_workers=0, drop_last=True)
    print('==============================================')
    print("start training Discriminator")
    Disoptimizer = torch.optim.AdamW(Dismodel.parameters(), lr=args.lr, weight_decay=args.wd)
    Memoptimizer = torch.optim.AdamW(Memmodel.parameters(), lr=args.lr, weight_decay=args.wd)

    dis_loss_dict = []
    number_batch = 0

    for _, train_data in enumerate(data_loder):
        number_batch = number_batch + 1
        userID = train_data[0].clone().unsqueeze(1)
        userID = userID.expand(args.bs, args.num_pnitems)
        userID = userID.cuda()
        itemID = train_data[1].clone()
        itemID = itemID.cuda()
        r_real = train_data[2].clone()
        r_real = r_real.cuda()

        Disoptimizer.zero_grad()
        Memoptimizer.zero_grad()
        # Forward pass
        r_gen = Genmodel(userID, itemID)
        r_mem = Memmodel(userID, itemID)
        pred_real = Dismodel(r_mem, r_real)
        pred_gen = Dismodel(r_mem, r_gen)

        # Wgan
        dis_loss_real = -torch.mean(pred_real)
        dis_loss_fake = torch.mean(pred_gen)
        dis_loss = dis_loss_real + dis_loss_fake
        # Backward and optimize
        dis_loss.backward()

        Disoptimizer.step()
        Memoptimizer.step()
        # print information and save loss
        dis_loss_dict.append(dis_loss.item())
        for p in Dismodel.parameters():
            p.data.clamp_(-args.c, args.c)

    dis_loss_dict = torch.tensor(dis_loss_dict)
    dis_loss_epoch = torch.mean(dis_loss_dict)
    dis_loss_epoch = dis_loss_epoch.item()

    return dis_loss_epoch, Memmodel, Dismodel


def trainGAN(args, Genmodel, Memmodel, Dismodel, trainMat, trainList, user_train_dict):
    dataset = Dataset(trainMat, trainList, user_train_dict, args)  # number of tarining data record
    data_loder = data.DataLoader(dataset=dataset, batch_size=args.bs, shuffle=True, num_workers=0, drop_last=True)
    Disoptimizer = torch.optim.AdamW(Dismodel.parameters(), lr=args.lr, weight_decay=args.wd)
    Memoptimizer = torch.optim.AdamW(Memmodel.parameters(), lr=args.lr, weight_decay=args.wd)
    Genoptimizer = torch.optim.AdamW(Genmodel.parameters(), lr=args.lr, weight_decay=args.wd)

    dis_loss_dict = []
    gen_loss_dict = []
    number_batch = 0

    for _, train_data in enumerate(data_loder):
        number_batch = number_batch + 1
        userID = train_data[0].clone().unsqueeze(1)
        userID = userID.expand( args.bs, args.num_pnitems)
        userID = userID.cuda()
        itemID = train_data[1].clone()
        itemID = itemID.cuda()
        r_real = train_data[2].clone()
        r_real = r_real.cuda()


        Disoptimizer.zero_grad()
        Memoptimizer.zero_grad()
        Genoptimizer.zero_grad()

        # Forward pass
        r_gen = Genmodel(userID, itemID)
        r_mem = Memmodel(userID, itemID)
        pred_real = Dismodel(r_mem, r_real)
        pred_gen = Dismodel(r_mem, r_gen)

        # culculate loss D_fake,D_real
        dis_loss_real = -torch.mean(pred_real)
        dis_loss_fake = torch.mean(pred_gen)
        dis_loss = dis_loss_real + dis_loss_fake

        # Backward and optimize
        dis_loss.backward()
        Disoptimizer.step()
        Memoptimizer.step()
        for p in Dismodel.parameters():
            p.data.clamp_(-args.c, args.c)

        # print information and save loss
        dis_loss_dict.append(dis_loss.item())

        Genoptimizer.zero_grad()
        r_gen = Genmodel(userID, itemID)
        r_mem = Memmodel(userID, itemID)
        pred_gen = Dismodel(r_mem ,r_gen)
        gen_loss = -torch.mean(pred_gen)
        gen_loss.backward()

        Genoptimizer.step()
        
        # print information and save loss
        gen_loss_dict.append(gen_loss.item())

    dis_loss_dict = torch.tensor(dis_loss_dict)
    dis_loss_epoch = torch.mean(dis_loss_dict)
    dis_loss_epoch = dis_loss_epoch.item()

    gen_loss_dict = torch.tensor(gen_loss_dict)
    gen_loss_epoch = torch.mean(gen_loss_dict)
    gen_loss_epoch = gen_loss_epoch.item()

    return gen_loss_epoch, Genmodel, dis_loss_epoch, Memmodel, Dismodel


def testmodel( Genmodel, testList, evalItems, topk):
    num_data = len(testList)
    hits_dict = []
    ndcg_dict = []
    with torch.no_grad():
        for idx in range(num_data):
            user = testList[idx][0]
            gtItem = testList[idx][1]
            items = evalItems[user].copy()
            items.append(gtItem)
            map_item_score = {}
            
            # Convert numpy arrays to torch tensors
            userID = torch.full((1,100), user, dtype=torch.long)
            userID = userID.cuda()
            itemID = torch.tensor(items, dtype=torch.long)
            itemID = torch.unsqueeze(itemID,dim=0)
            itemID = itemID.cuda()

            # Forward pass; predict using Genmodel
            predict =  Genmodel(userID,itemID)

            for i in range(len(items)):
                item = int(items[i])
                map_item_score[item] = predict[0,i].item()

            ranklist = heapq.nlargest(topk, map_item_score, key=map_item_score.get)
            hr = _getHitRatio(ranklist, gtItem)
            ndcg = _getNDCG(ranklist, gtItem)
            hits_dict.append(hr)
            ndcg_dict.append(ndcg)
            

        hits_dict = torch.tensor(hits_dict)
        averagehr = torch.mean(hits_dict).item()
        ndcg_dict = torch.tensor(ndcg_dict)
        averagendcg = torch.mean(ndcg_dict).item()
        
    return averagehr, averagendcg


def _getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1.
    return 0.


def _getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0.