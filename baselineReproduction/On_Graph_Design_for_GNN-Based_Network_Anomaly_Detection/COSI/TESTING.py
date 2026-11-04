import numpy as np
from models.GDN import GDN
from test  import test
from util.TimeDataset import TimeDataset
import torch
from torch.utils.data import DataLoader, random_split, Subset
import random
import matplotlib.pyplot as plt
import pandas as pd
from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data

from util.TimeDataset import TimeDataset


from models.GDN import GDN

from util.train import train
from test  import test
from util.evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores


#my adjacency matrix of the graph structure in csv format
adj_25=np.genfromtxt('/home/student/Documents/Data/Training_Set_Day_1/adj25d_th09.csv', delimiter=',')
#train
df_train=pd.read_csv('//home/killian/Documents/Data/results/label_full_mat_pck_loss.csv', skiprows=range(1,3000), usecols=range(1,20), nrows=10000)
columns_train=df_train.columns
#val
df_val=pd.read_csv('/home/student/Documents/Data/Training_Set_Day_8/dataset_end_total_preprocessed.csv', nrows=100000)
columns_val=df_val.columns
#test
df_test=pd.read_csv('/home/student/Documents/Data/40/physical_level/dataset_test_full_server_1_preprocessed.csv', nrows=100000)
columns_test=df_test.columns



print(df_val.shape)
print(df_train.shape)
print(df_test.shape)
columns_train=df_train.columns
columns_val=df_val.columns
columns_test=df_test.columns

def fil_edges(Mat):
    edges_index=[]
    X1=[]
    X2=[]
    n=len(Mat)
    for k in range(len(Mat)):
        for j in range(len(Mat[0])):
            if k+j+1<n:
                if Mat[j][j+k+1]>0:
                    X1.append(j)
                    X2.append(j+k+1)
    edges_index.append(X1)
    edges_index.append(X2)
    edges_index=torch.tensor(edges_index)
    return edges_index


batch=128
epoch=20
slide_win=30  
dim=64
slide_stride=5
comment=''
random_seed=0
out_layer_num=1
out_layer_inter_dim=256
decay=0 
val_ratio=0.1 
topk=20 #not used except for random tasks
dataset='5G3E'
report = 'best'
device='cpu' #'cida' if GPU available but here CPU
load_model_path=''


train_config = {
    'batch': batch,
    'epoch': epoch,
    'slide_win': slide_win,
    'dim': dim,
    'slide_stride': slide_stride,
    'comment': comment,
    'seed': random_seed,
    'out_layer_num': out_layer_num,
    'out_layer_inter_dim': out_layer_inter_dim,
    'decay': decay,
    'val_ratio': val_ratio,
    'topk': topk,
}

env_config={
    'save_path': '',
    'dataset': dataset,
    'report': report,
    'device': device,
    'load_model_path': load_model_path
}



def get_loaders( train_dataset, seed, batch, val_ratio=0.1):
    dataset_len = int(len(train_dataset))
    train_use_len = int(dataset_len * (1 - val_ratio))
    val_use_len = int(dataset_len * val_ratio)
    val_start_index = random.randrange(train_use_len)
    indices = torch.arange(dataset_len)

    train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
    train_subset = Subset(train_dataset, train_sub_indices)

    val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
    val_subset = Subset(train_dataset, val_sub_indices)


    train_dataloader = DataLoader(train_subset, batch_size=batch,
                            shuffle=True)

    val_dataloader = DataLoader(val_subset, batch_size=batch,
                            shuffle=False)

    return train_dataloader, val_dataloader

   

def get_score( test_result, val_result):

    feature_num = len(test_result[0][0])
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)

    test_labels = np_test_result[2, :, 0].tolist()

    test_scores, normal_scores = get_full_err_scores(test_result, val_result)

    top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1) 
    top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)


    print('=========================** Result **============================\n')

    info = None
    if env_config['report'] == 'best':
        info = top1_best_info
    elif env_config['report'] == 'val':
        info = top1_val_info

    print(f'F1 score: {info[0]}')
    print(f'precision: {info[1]}')
    print(f'recall: {info[2]}')



edges_index=fil_edges(adj_25)
print(len(edges_index))
print(len(edges_index[0]))
print(edges_index[0])

fc_edge_index = torch.tensor(edges_index, dtype = torch.long)
feature_map=df_train.columns
edge_index_sets = []
edge_index_sets.append(fc_edge_index)

cfg = {
    'slide_win': train_config['slide_win'],
    'slide_stride': train_config['slide_stride'],
}

input_data_test = df_test  
input_data_val=df_val

graphs=edges_index
test_dataset_indata = construct_data(input_data_test, feature_map, labels=input_data_test.attack.tolist())
test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)
test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                            shuffle=False, num_workers=0)



val_dataset_indata = construct_data(input_data_val, feature_map, labels=0)
val_dataset = TimeDataset(val_dataset_indata, fc_edge_index, mode='test', config=cfg)
val_dataloader = DataLoader(val_dataset, batch_size=train_config['batch'],
                            shuffle=False, num_workers=0)


print('data construction over')


model=GDN(edge_index_sets, len(feature_map), 
                dim=train_config['dim'], 
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk']
            ).to(device)

model.load_state_dict(torch.load('pretrained/best_07|15-11:07:55.pt'))
model.eval()
print('model loaded')

best_model = model.to(device)
avg_loss, table_result = test(best_model, test_dataloader)
val_result = test(best_model, val_dataloader)
val_avg, table_val=val_result[0], val_result[1]
get_score(table_result, table_val)