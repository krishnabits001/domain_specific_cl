import sys

def train_data(no_of_tr_imgs,comb_of_tr_imgs):
    #print('train set list')
    if(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c1'):
        labeled_id_list=["1003"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c2'):
        labeled_id_list=["1006"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c3'):
        labeled_id_list=["1008"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c4'):
        labeled_id_list=["1001"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c5'):
        labeled_id_list=["1009"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c1'):
        labeled_id_list=["1001","1003"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c2'):
        labeled_id_list=["1009","1008"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c3'):
        labeled_id_list=["1006","1002"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c4'):
        labeled_id_list=["1010","1007"]
    # Use 'tr8' list to train the Benchmark/Upperbound
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c1'):
        labeled_id_list=["1001","1003","1004","1006","1007","1008","1009","1010"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c2'):
        labeled_id_list=["1002","1003","1004","1005","1006","1007","1009","1010"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c3'):
        labeled_id_list=["1001","1003","1004","1005","1006","1008","1009","1010"]
    elif(no_of_tr_imgs=='tr10'):
        # Use this list of subject ids as unlabeled data during pre-training
        labeled_id_list=["1001","1002","1003","1004","1005","1006","1007","1008","1009","1010"]
    else:
        print('Error! Select valid combination of training images')
        sys.exit()
    return labeled_id_list

def val_data(no_of_tr_imgs,comb_of_tr_imgs):
    #print('val set list')
    if(no_of_tr_imgs=='tr1' and (comb_of_tr_imgs=='c1' or comb_of_tr_imgs=='c4')):
        val_list=["1002","1005"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c2'):
        val_list=["1001","1008"]
    elif(no_of_tr_imgs=='tr1' and (comb_of_tr_imgs=='c3' or comb_of_tr_imgs=='c5')):
        val_list=["1007","1002"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c1'):
        val_list=["1002","1005"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c2'):
        val_list=["1007","1002"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c3'):
        val_list=["1001","1008"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c4'):
        val_list=["1004","1008"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c1'):
        val_list=["1002","1005"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c2'):
        val_list=["1001","1008"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c3'):
        val_list=["1007","1002"]
    elif(no_of_tr_imgs=='tr10'):
        val_list=["1002","1005","1008","1010"]
    return val_list

def test_data():
    #print('test set list')
    test_list=["1011","1012","1013","1014","1015","1016","1017","1018","1019","1020"]
    return test_list
