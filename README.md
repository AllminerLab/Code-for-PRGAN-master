Some tips to use this package code:


## Environment Settings
We use Pytorch as the backend. 
- torch version: '1.3.0'
- numpy version: '1.17.0'
- scipy version: '1.5.2'
- cudnn version: '7.6.5'

### Dataset
We provide all four processed datasets: MovieLens 1 Million (ml-1m), LastFm (lastfm) and Yelp. 

1.**traindata.csv**
- Train file.
- Each Line is a training instance: `userID\t itemID\t rating\t timestamp (if have)`

2.**testdata.csv**
- Test file (positive instances). 
- Each Line is a testing instance: `userID\t itemID\t rating\t timestamp (if have)`

3.**testnegative.csv**
- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 99 negative samples.  
- Each line is in the format: `(userID,itemID)\t negativeItemID1\t negativeItemID2 ...`



## Example to run the codes.
The instruction of commands has been clearly stated in the codes (see the  parse_args function). 

Run with ml1m dataset:
```
python main_prmf.py --data_path data/ml1m/ --valid_record 3. --num_positems 2 --num_pnitems 10 --c 0.006
```

Run with lastfm dataset:
```
python main_prmf.py --data_path data/lastfm/ --valid_record 0. --num_positems 2 --num_pnitems 13 --c 0.006
```

Run with yelp dataset:
```
python main_prmf.py --data_path data/yelp/ --valid_record 3. --num_positems 1 --num_pnitems 11 --c 0.0064
```

Run with pinterest dataset:
```
python main_prmf.py --data_path data/pinterest/ --valid_record 0. --num_positems 2 --num_pnitems 11 --c 0.0062
```


## The tunable parameters are listed as follows:
1. data_path: Data path.
2. valid_record: Records greater than the threshold are seen as valid interactions.
3. num_positems: Number of positive items in traindata. \p
4. num_pnitems: Number of items in traindata. \k
5. c: Weight clipping for Discriminator. \c
6. lr: Learning rate for parameters.
7. wd: Weight decay for parameters.


## References:
Jing Wen, Bi-Yi Chen, Chang-Dong Wang, and Zhihong Tian, “PRGAN: Personalized Recommendation with Conditional Generative Adversarial Networks.” ICDM 2021.

