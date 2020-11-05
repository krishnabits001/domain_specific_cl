import sys

def train_data(no_of_tr_imgs,comb_of_tr_imgs):
    #print('train set list')
    if(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c1'):
        labeled_id_list=["001"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c2'):
        labeled_id_list=["002"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c3'):
        labeled_id_list=["006"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c4'):
        labeled_id_list=["000"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c5'):
        labeled_id_list=["004"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c1'):
        labeled_id_list=["000","001"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c2'):
        labeled_id_list=["021","024"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c3'):
        labeled_id_list=["001","002"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c4'):
        labeled_id_list=["004","010"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c5'):
        labeled_id_list=["000","007"]
    elif (no_of_tr_imgs == 'tr22'):
        # Use this list of subject ids as unlabeled data during pre-training
        labeled_id_list=["000","001","002","004","006","007","010","013","014","016",\
                           "017","018","020","021","024","025","028"]
    elif (no_of_tr_imgs == 'trall'):
        # Use this list to train the Benchmark/Upperbound
        labeled_id_list=["000" "001","002","004","006","007","010","016","017","018", \
                           "020","021","024","025","028"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c1'):
        labeled_id_list=["000","001","002","004","006","007","010","016"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c2'):
        labeled_id_list=["002","004","006","007","010","016","021","018"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c3'):
        labeled_id_list=["000","001","006","007","010","017","018","024"]
    else:
        print('Error! Select valid combination of training images')
        sys.exit()
    return labeled_id_list

def val_data(no_of_tr_imgs,comb_of_tr_imgs):
    #print('val set list')
    if(no_of_tr_imgs=='tr1' and (comb_of_tr_imgs=='c1' or comb_of_tr_imgs=='c4')):
        val_list=["013","014"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c2'):
        val_list=["017","020"]
    elif(no_of_tr_imgs=='tr1' and (comb_of_tr_imgs=='c3' or comb_of_tr_imgs=='c5')):
        val_list=["025","028"]
    elif(no_of_tr_imgs=='tr2' and (comb_of_tr_imgs=='c1' or comb_of_tr_imgs=='c4')):
        val_list=["013","014"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c2'):
        val_list=["017","020"]
    elif(no_of_tr_imgs=='tr2' and (comb_of_tr_imgs=='c3' or comb_of_tr_imgs=='c5')):
        val_list=["025","028"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c1'):
        val_list=["013","014"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c2'):
        val_list=["017","020"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c3'):
        val_list=["025","028"]
    elif(no_of_tr_imgs=='tr22'):
        val_list=["013","014","020","028"]
    elif (no_of_tr_imgs == 'trall'):
        val_list=["013","014"]
    return val_list

def test_data():
    #print('test set list')
    test_list=["029","031","032","034","035","037","038","039","040","041", \
               "042","043","044","046","047"]
    return test_list
