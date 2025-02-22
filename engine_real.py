import torch
import torch.nn.functional as F
from prettytable import PrettyTable
import numpy as np
import random
from sklearn import metrics
import wandb
from tqdm import tqdm
from utils import *

def normalized_laplacian(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  D^-1/2 * ( D - A ) * D^-1/2 = I - D^-1/2 * ( A ) * D^-1/2
    """
    out_degree = np.array(A.sum(1), dtype=np.float32)
    int_degree = np.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.eye(A.shape[0]) - np.diag(out_degree_sqrt_inv) @ A @ np.diag(int_degree_sqrt_inv)
    return mx_operator

def process_instance_faketopo(A, numericals, r_truth, args):
    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    A = np.array(A)
    add0 = np.ones((1, A.shape[0]))#[1,3]
    add1 = np.zeros((A.shape[0]+1, 1))#[4,1]
    A = np.concatenate((A, add0), axis=0)#[4,3]
    A = np.concatenate((A, add1), axis=1)#[4,4]

    A = normalized_laplacian(A)

    A = torch.from_numpy(A).to(torch.float32).to(device)

    numericals = np.array(numericals)
    numericals = np.transpose(numericals, (0,2,1))#[num,3,11]
    add_nume = np.mean(numericals, axis=1, keepdims=True)#keepdim保持维数不变[num,1,11]
    numericals = np.concatenate((numericals, add_nume), axis=1)#[num,4,11]
    numericals = torch.from_numpy(numericals).to(torch.float32).to(device)
    r_truth = torch.from_numpy(np.array(r_truth)).to(torch.float32).to(device)

    return A, numericals, r_truth


def train_test_faketopo(fold, train_ids, test_ids,model, train_subset, test_subset, train_loader, test_loader, optimizer, criterion, args):

    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    if args.use_wandb:

        dir = wandb.run.dir
    else:
        dir = os.path.dirname(os.path.abspath(__file__))

    make_model_dirs(dir)

    checkpoint_saver = CheckpointSaver(dirpath=os.path.join(dir, 'checkpoints'), decreasing=False, top_n=1, not_save = (args.use_model != 'resinf'))

    metric_monitor = MetricMonitor()

    for epoch in range(args.epoch):

        model.train()
        total_loss = 0
        for i, (A, numericals, r_truth) in tqdm(enumerate(train_loader), total=len(train_subset) // args.train_size + 1):#49train
            A = A.numpy()
            A = A.squeeze(0)
            numericals = numericals.numpy()
            numericals = numericals.squeeze(0)#去除size为1的维度，0代表第一维度，1第二维度
            A, numericals, r_truth = process_instance_faketopo(A, numericals, r_truth, args)#[4,4],[num,4,11]
            numericals_use = numericals.index_select(0, torch.tensor(random.choices(list(range(numericals.shape[0])), k=args.K)).to(device))#[2,4,11]
            # if args.extra_dim:#[2,4or7or13or25,1+hidden2=3-4轨迹数]?
            r_pred, _, __  = model(numericals_use[:,:,:1+args.hidden], A)

            # else:
            #     r_pred, _, __  = model(numericals_use[:,:,1:1+args.hidden], A)

            if not torch.isnan(r_pred):
                loss = criterion(r_pred, r_truth)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        total_loss = total_loss / len(train_subset)
        print('Total Loss in Epoch {0}:'.format(epoch))
        print(total_loss)
        if args.use_wandb:
            wandb.log({'train_loss': total_loss, "epoch": epoch})

        r_pred = min(r_pred + 1e-6, torch.FloatTensor([1]).to(device))

        with torch.no_grad():

            test_loss = 0

            preds = []
            truths = []
            pred_labels = []

            for i, (A, numericals, r_truth) in tqdm(enumerate(test_loader), total=len(test_subset) // args.test_size + 1):#7test
                

                A = A.numpy()
                A = A.squeeze(0)
                numericals = numericals.numpy()
                numericals = numericals.squeeze(0)

                A, numericals, r_truth = process_instance_faketopo(A, numericals, r_truth, args)
                numericals_use = numericals.index_select(0, torch.tensor(random.choices(list(range(numericals.shape[0])), k=args.K)).to(device))
                # if args.extra_dim:
                r_pred, _, __  = model(numericals_use[:,:,:1+args.hidden], A)
                if torch.isnan(r_pred):
                    print('fold',fold,'epoch',epoch,'train_ids',train_ids,'test_ids',test_ids)
                    print('shape', numericals_use[:, :, :1 + args.hidden].shape, A.shape)
                    #print('why',numericals_use[:,:,:1+args.hidden],A)
                    print('i', i, 'r_pred', r_pred, 'r_truth', r_truth)
                # else:
                #     r_pred, _, __  = model(numericals_use[:,:,1:1+args.hidden], A)
                r_pred = min(r_pred + 1e-6, torch.FloatTensor([1]).to(device))

                if not torch.isnan(r_pred):
                    
                    loss = criterion(r_pred, r_truth)
                    test_loss += loss.item()
                    preds.append(r_pred.item())
                    truths.append(r_truth.item())
                    pred_labels.append((r_pred.item() > args.threshold))

            test_loss = test_loss / len(test_subset)

            print('truths',truths,'preds',preds)
            my_auc = metrics.roc_auc_score(truths, preds)
            my_f1 = metrics.f1_score(truths, pred_labels, average='weighted')
            my_acc = metrics.accuracy_score(truths, pred_labels)
            my_mcc = metrics.matthews_corrcoef(truths, pred_labels)

            checkpoint_saver(model, epoch, my_f1)
            metric_monitor.update(my_f1, my_acc, my_mcc, my_auc, epoch)

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "Train Loss", "Test Loss", "Accuracy", "AUC", "f1", "mcc", "Positive"]
            train_res.add_row([epoch, total_loss, test_loss, my_acc, my_auc, my_f1, my_mcc, sum(truths)/len(truths)])
            print(train_res)
            if args.use_wandb:
                wandb.log({'test_loss': test_loss, "epoch": epoch, "Acc": my_acc, "AUC": my_auc, "F1": my_f1, "Mcc": my_mcc, "positive": sum(truths)/len(truths)})

    f1, acc, mcc, auc, epoch = metric_monitor.read()

    return [f1, acc, mcc, auc, epoch]



















    


        
        





    
    





