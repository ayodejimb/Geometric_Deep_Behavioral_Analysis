import torch
import math
import numpy as np
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry.core as tgmc
import copy
from processing.CenteredScaled import CenteredScaled
from processing.inv_exp import inv_exp
from processing.PT import *
from models.RigidNet import RigidNet
from models.NonRigidNet import NonRigidNet
import json
np.random.seed(42)

all_dist = {'within': [35,612], 'beyond': [35,1739], 'combine': [35,1840]} # epoch and LSTM parameters

def load_data(cat):
    
    file_path_data = r'C:\Downloads\Multi_Class_OiP\{}\all_data.npy'.format(cat)  # for OR, use its path 
    file_path_label = r'C:\Downloads\Multi_Class_OiP\{}\all_label.pickle'.format(cat)
    all_data = np.load(file_path_data,  allow_pickle=True)
    all_label = np.load(file_path_label,  allow_pickle=True)

    return all_data, all_label

cuda = torch.cuda.is_available()
device = 'cuda' if cuda == True  else 'cpu'
if device == 'cuda':
    print('Using CUDA')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()
else:
    print('NOT using CUDA')

options=['RigidTransform','NonRigidTransform','RigidTransformInit','NonRigidTransformInit']

# ========== First Big Loop to extract all samples ==============
for opt, data in all_dist.items():    
    data_samples, label_samples = load_data(opt)
    print(data_samples.shape, len(label_samples[1]))

    res = {}
    max_acc_per_fold = {}

    # **** for the attention weights ****
    folds_chan_att_w = {}
    folds_spa_att_w = {}
    folds_chan_att_r = {}
    folds_spa_att_r = {}
    folds_data_last_batch = {}
    folds_label_last_batch = {}

    # path to save the attention weights and clasification results
    save_att = r'C:\Desktop\attention_multi_class\both_attn_results_OiP\{}'.format(opt)
    os.makedirs(save_att, exist_ok=True)
    file_path = r"C:\Downloads\Multi_Class_OiP\{}\classsification_attn_results.json".format(opt)

    # ============= The LOOCV Loop ==================
    indexes = []
    for index in range(data_samples.shape[0]):

        X_train = np.delete(data_samples, index, axis=0)
        X_test = data_samples[index:index+1]

        y_train = label_samples[1][:index] + label_samples[1][index+1:]
        y_test = label_samples[1][index]

        training_epochs = data[0]
        batch_size= 8
        batch_size_test = 1
        learning_rate=0.0001
        runs = 10
        save_max = True
        cast = 'log_0refseq'

        print('Running ', opt,  ' for fold ', index+1)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]//2 , 2))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]//2 , 2))
        y_train = np.array(y_train).astype('int32')
        y_test = np.array(y_test).astype('int32')
        print('Label Test is ', y_test)

        ref_skel = copy.deepcopy(X_train[0,0])

        for i in range(X_train.shape[0]): 
            for j in range(X_train[i].shape[0]): 
                X_train[i, j, 0:8] = CenteredScaled(X_train[i, j, 0:8])

        for i in range(X_test.shape[0]):
            for j in range(X_test[i].shape[0]):
                X_test[i, j, 0:8] = CenteredScaled(X_test[i, j, 0:8]) 

        if cast == 'log_sref':
            for i in range(X_train.shape[0]):
                for j in range(X_train[i].shape[0]):
                    try:
                        X_train[i, j, 0:8] = inv_exp(ref_skel[0:8], X_train[i, j, 0:8])
                    except:
                        i = i + 1
            for i in range(X_test.shape[0]):
                for j in range(X_test[i].shape[0]):
                    try:
                        X_test[i, j, 0:8] = inv_exp(ref_skel[0:8], X_test[i, j, 0:8])
                    except:
                        i = i + 1     

        elif cast == 'log_0refseq':
            for i in range(X_train.shape[0]):
                for j in range(X_train[i].shape[0]):
                    try:
                        X_train[i, j, 0:8] = inv_exp(X_train[i, 0, 0:8], X_train[i, j, 0:8])
                    except:
                        i = i + 1 
            for i in range(X_test.shape[0]):
                for j in range(X_test[i].shape[0]):
                    try:
                        X_test[i, j, 0:8] = inv_exp(X_test[i, 0, 0:8], X_test[i, j, 0:8])
                    except:
                        i = i + 1
        else:
            print('Casting variant doesn\'t exist')
            quit()

        
        num_frames = X_train.shape[1]
        num_joints = X_train.shape[2]
        dims = X_train.shape[3]
        num_channels = num_joints * dims


        max_acc_per_trans = []

        # ******* for the attentions *******
        att_chan_w = []
        att_spat_w = []
        att_chan_r = []
        att_spat_r = []

        for m in options:
            loss = []
            mod = m
            if m == 'RigidTransform' or m == 'RigidTransformInit':
                rigid = True
            else:
                rigid = False

            acc = []

            # ******* for the attentions *******
            chan_att_w = []
            spa_att_w = []
            chan_att_r = []
            spa_att_r = []

            for r in range(runs):

                if rigid == True:
                    model = RigidNet(mod,num_frames,num_joints, data[1], r).to(device)
                else:
                    model = NonRigidNet(mod,num_frames,num_joints, data[1], r).to(device)
            
                criterion = nn.CrossEntropyLoss()

                opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
                steps = len(X_train) // batch_size

                model.train()
                for epoch in range(training_epochs):
                    correct=0
                    total=0
                    epoch_loss = 0.0

                    for i in range(steps + (1 if len(X_train) % batch_size != 0 else 0)):                       
                        start_idx = i * batch_size
                        end_idx = start_idx + batch_size
                        x, y = X_train[start_idx:end_idx], y_train[start_idx:end_idx]
                        if end_idx > len(X_train):
                            x = X_train[start_idx:]
                            y = y_train[start_idx:]
                        inputs, labels = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
                        opt.zero_grad()   
                        output = model(inputs.float())

                        loss = criterion(output, labels.long())
                        loss.backward()
                        opt.step()
                        epoch_loss += loss.item()
                        y_pred_softmax = torch.log_softmax(output.data, dim = 1)
                        _, predicted = torch.max(y_pred_softmax, 1)
                        total += labels.size(0)
                        correct += (predicted == labels.long()).sum().item()

                    accuracy = 100 * correct / total
                    epoch_loss = epoch_loss / len(X_train)

                correct_test = 0
                total_test = 0

                model.eval()
                with torch.no_grad():
                    steps = int(len(X_test) / batch_size_test)
                    for i in range(steps):                        
                        x, y = X_test[i*batch_size_test:(i*batch_size_test)+batch_size_test], y_test                
                        inputs, labels = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
                        outputs = model(inputs.float())

                        y_pred_softmax = torch.log_softmax(outputs.data, dim = 1)
                        _, predicted = torch.max(y_pred_softmax, 1)
                        total_test += 1
                        correct_test += (predicted == labels.long()).sum().item()

                        # *** get the attention weights and results for the test set 
                        ch_w = model.att.attention_maps['channel_weights']
                        sp_w = model.att.attention_maps['spatial_weights']
                        ch_r = model.att.attention_maps['channel_results']
                        sp_r = model.att.attention_maps['spatial_results']

                # === Append the acc, channel and spatial attention map for each run (saved for Test set)
                chan_att_w.append(ch_w)
                spa_att_w.append(sp_w)
                chan_att_r.append(ch_r)
                spa_att_r.append(sp_r)

                accuracy = 100*correct_test/total_test

                # Append the accuracy for each run here 
                acc.append(accuracy)

            max_acc_per_trans.append(max(acc))
            print('Max acc for {} is {}'.format(mod, max(acc)))   

            # append the maximum accuracy out of all the 10 runs + its channel and spatial attention map (each of this list below has len of 4 (rigid, rigidinit...))
            max_index = acc.index(max(acc))
            att_chan_w.append(chan_att_w[max_index])
            att_spat_w.append(spa_att_w[max_index])
            att_chan_r.append(chan_att_r[max_index])
            att_spat_r.append(spa_att_r[max_index])
        
        max_acc_per_fold[index+1] = max_acc_per_trans
        print('The max accs for fold {} is {} '.format(index+1, max_acc_per_trans))

        indexes.append(index)

        # Save the accuarcy and attention maps for this fold in a dictionary. And also the test set of this fold and its label
        folds_chan_att_w[index+1] = att_chan_w
        folds_spa_att_w[index+1] = att_spat_w
        folds_chan_att_r[index+1] = att_chan_r
        folds_spa_att_r[index+1] = att_spat_r
        folds_data_last_batch[index+1] = X_test
        folds_label_last_batch[index+1] = y_test

    # Compute the average/transform for each fold and retun the highest
    columns = list(zip(*max_acc_per_fold.values()))
    averages = [sum(col) / len(col) for col in columns]
    max_avg = max(averages)
    max_column = averages.index(max_avg)
    print("Max average is {} with value {:.2f}".format(options[max_column], max_avg))

    # ============  Then finally save in a dictionary ==============
    res[opt] = (indexes, columns[max_column])

    # Save the dictionary acc to a file
    with open(file_path, "w") as f:
        json.dump(res, f)

    # *** Finally, save the attention maps of this model for each fold
    for (fold_chan, chan_attn_w),(fold_spa, spa_attn_w), (fold_chan_, chan_attn_r), (fold_spa_, spa_attn_r), (fold_data, data_last_batch), (fold_label, label_last_batch) in zip (folds_chan_att_w.items(), folds_spa_att_w.items(), folds_chan_att_r.items(), folds_spa_att_r.items(), folds_data_last_batch.items(),folds_label_last_batch.items()):

        assert fold_chan == fold_spa == fold_chan_ == fold_spa_ == fold_data == fold_label, "Mismatch in fold names between channel and spatial attention maps"
        
        fold_path = os.path.join(save_att, str(fold_chan))
        os.makedirs(fold_path, exist_ok=True)

        np.save(os.path.join(fold_path, 'chan_att_w.npy'), chan_attn_w[max_column])
        np.save(os.path.join(fold_path, 'spa_att_w.npy'), spa_attn_w[max_column])
        np.save(os.path.join(fold_path, 'chan_att_r.npy'), chan_attn_r[max_column])
        np.save(os.path.join(fold_path, 'spa_att_r.npy'), spa_attn_r[max_column])
        np.save(os.path.join(fold_path, 'data.npy'), data_last_batch)
        np.save(os.path.join(fold_path, 'label.npy'), label_last_batch)