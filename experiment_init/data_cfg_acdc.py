import sys

def train_data(no_of_tr_imgs,comb_of_tr_imgs):
    #print('train set list')
    if(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c1'):
        labeled_id_list=["002"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c2'):
        labeled_id_list=["042"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c3'):
        labeled_id_list=["095"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c4'):
        labeled_id_list=["022"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c5'):
        labeled_id_list=["062"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c1'):
        labeled_id_list=["042","062"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c2'):
        labeled_id_list=["002","042"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c3'):
        labeled_id_list=["042","095"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c4'):
        labeled_id_list=["002","022"]
    elif(no_of_tr_imgs=='tr2' and comb_of_tr_imgs=='c5'):
        labeled_id_list=["002","095"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c1'):
        labeled_id_list=["002","022","042","062","095","003","023","043"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c2'):
        labeled_id_list=["002","022","042","062","095","063","094","043"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c3'):
        labeled_id_list=["002","022","042","062","095","003","094","023"]
    elif(no_of_tr_imgs=='tr52'):
        # Use this list of subject ids as unlabeled data during pre-training
        labeled_id_list=["001","002","003","004","005","006","017","018","019","020","012",\
                         "021","022","023","024","025","026","037","038","039","040", \
                         "041","042","043","044","045","046","057","058","059","060",\
                         "061","062","063","064","065","066","077","078","079","080","072",\
                         "081","082","083","084","085","086","097","098","099","100"]
    elif (no_of_tr_imgs=='trall'):
        # Use this list to train the Benchmark/Upperbound
        labeled_id_list = ["001","002","003","004","005","006","012","013","014","015","016","017","018","019","020", \
                           "021","022","023","024","025","026","011","032","033","034","035","036","037","038","039","040", \
                           "041","042","043","044","045","046","051","052","053","054","055","056","057","058","059","060", \
                           "061","062","063","064","065","066","072","073","074","075","076","077","078","079","080", \
                           "081","082","083","084","085","086","091","092","093","094","095","096","097","098","099","100" ]
    else:
        print('Error! Select valid combination of training images')
        sys.exit()
    return labeled_id_list

def val_data(no_of_tr_imgs,comb_of_tr_imgs):
    #print('val set list')
    if(no_of_tr_imgs=='tr1' and (comb_of_tr_imgs=='c1' or comb_of_tr_imgs=='c5')):
        val_list=["011","071"]
    elif(no_of_tr_imgs=='tr1' and (comb_of_tr_imgs=='c2')):
        val_list=["031","072"]
    elif(no_of_tr_imgs=='tr1' and (comb_of_tr_imgs=='c3' or comb_of_tr_imgs=='c4')):
        val_list=["011","071"]
    elif(no_of_tr_imgs=='tr2' and (comb_of_tr_imgs=='c2')):
        val_list=["011","071"]
    elif(no_of_tr_imgs=='tr2' and (comb_of_tr_imgs=='c1' or comb_of_tr_imgs=='c3')):
        val_list=["031","072"]
    elif(no_of_tr_imgs=='tr2' and (comb_of_tr_imgs=='c4' or comb_of_tr_imgs=='c5')):
        val_list=["011","071"]
    elif(no_of_tr_imgs=='tr8' and (comb_of_tr_imgs=='c1' or comb_of_tr_imgs=='c3')):
        val_list=["011","071"]
    elif(no_of_tr_imgs=='tr8' and comb_of_tr_imgs=='c2'):
        val_list=["031","072"]
    elif(no_of_tr_imgs=='tr52'):
        val_list=["013","014","033","034","053","054","073","074","093","094"]
    elif(no_of_tr_imgs=='trall'):
        val_list=["011","071"]
    return val_list

def test_data():
    #print('test set list')
    test_list=["007","008","009","010", "027","028","029","030", \
               "047","048","049","050", "067","068","069","070", "087","088","089","090"]
    return test_list
