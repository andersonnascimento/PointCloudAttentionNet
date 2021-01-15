#!/usr/bin/env python

from utils.params import Params

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.utility import calculate_loss

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.model_net_40 import ModelNet40
from dataset.model_net_10 import ModelNet10

# import model
from model.attention_dgcnn import AttentionDGCNN
from model.dgcnn import DGCNN
from model.point_attention_net import PointAttentionNet
from model.point_net import PointNet

from tensorboardX import SummaryWriter

import sklearn.metrics as metrics

def train(args):
    train_loader = DataLoader(args.dataset_loader(partition='train', num_points=args.num_points, random_state=args.random_state),
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(args.dataset_loader(partition='validation', num_points=args.num_points, random_state=args.random_state),
                                   num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = args.device
    model = params.model(params).to(params.device)
    args.log(str(model),False)
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!") 

    writer = SummaryWriter(args.output_dir)

    if args.optimizer == 'SGD':
        print(f"{str(params.model)} use SGD")
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        print(f"{str(params.model)} use Adam")
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = calculate_loss

    if args.last_checkpoint() != "":
        model.load_state_dict(torch.load(args.last_checkpoint()))

    global_best_loss, global_best_acc, global_best_avg_acc = 0, 0, 0
    for epoch in range(args.epochs):
        epoch_results = []
        ts = time.time()
        def train_batch():
            scheduler.step()
            ####################
            # Train
            ####################
            train_loss = 0.0
            count = 0.0
            model.train()
            train_pred = []
            train_true = []

            for data, label in train_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                opt.zero_grad()
                logits = model(data)

                loss = criterion(logits, label)

                loss.backward()
                opt.step()
                preds = logits.max(dim=1)[1]
                count += batch_size
                train_loss += loss.item() * batch_size
                train_true.append(label.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())

                if args.dry_ryn:
                    break

            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            return train_loss*1.0/count, train_true, train_pred

        train_loss, train_true, train_pred = train_batch()
        if args.dry_ryn:
            break

        ####################
        # Validation
        ####################
        with torch.no_grad():
            val_loss = 0.0
            count = 0.0
            model.eval()
            val_pred = []
            val_true = []

            # best_val_loss, best_val_acc, best_val_avg_acc = 0, 0, 0
            for data, label in validation_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits = model(data)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                val_loss += loss.item() * batch_size
                val_true.append(label.cpu().numpy())
                val_pred.append(preds.detach().cpu().numpy())

            val_true = np.concatenate(val_true)
            val_pred = np.concatenate(val_pred)
            val_acc = metrics.accuracy_score(val_true, val_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(val_true, val_pred)

            accuracy = metrics.accuracy_score(train_true, train_pred)
            balanced_accuracy = metrics.balanced_accuracy_score(train_true, train_pred)

            val_loss = val_loss*1.0/count

            args.csv(
                epoch,
                train_loss,
                accuracy,
                balanced_accuracy,
                val_loss,
                val_acc,
                avg_per_class_acc,
                time.time()-ts)

            writer.add_scalar('train/loss', loss, epoch)
            writer.add_scalar('train/accuracy', accuracy, epoch)
            writer.add_scalar('train/balanced_accuracy', balanced_accuracy, epoch)

            writer.add_scalar('valid/loss', val_loss, epoch)
            writer.add_scalar('valid/accuracy', val_acc, epoch)
            writer.add_scalar('valid/balanced_accuracy', avg_per_class_acc, epoch)

            torch.save(model.state_dict(), args.checkpoint_path())
            if avg_per_class_acc > global_best_avg_acc:
                global_best_loss, global_best_acc, global_best_avg_acc = val_loss*1.0/count, val_acc, avg_per_class_acc
                torch.save(model.state_dict(), args.best_checkpoint())

        torch.cuda.empty_cache()
    args.print_summary(global_best_loss, global_best_acc, global_best_avg_acc)

def test(args, state_dict=None):
    test_loader = DataLoader(args.dataset_loader(partition='test', num_points=args.num_points, random_state=args.random_state),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = args.device
    model = params.model(params).to(params.device)
    model = nn.DataParallel(model)

    if state_dict != None:
        model.load_state_dict(torch.load(state_dict))
    else:
        model.load_state_dict(torch.load(params.best_checkpoint()))

    with torch.no_grad():
        model = model.eval()
        test_acc = 0.0
        count = 0.0
        test_true = []
        test_pred = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        if state_dict == None:
            outstr = 'TEST:: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
            args.log('====================================================================')
            args.log(outstr)

    return test_acc, avg_per_class_acc

if __name__ == "__main__":
    # params=Params(model=PointNet, epochs=5, num_points=1024, emb_dims=1024, k=20, optimizer='SGD', lr=0.0001, att_heads=8, momentum=0.9, dropout=0.5, dump_file=True, dry_run=False)
    # train(params)
    # test(params)
    hyperparams= [
        # { "optimizer": 'ADAM', "lr": 0.00001, "att_heads": 8 },
        # { "optimizer": 'ADAM', "lr": 0.0001, "att_heads": 8 },
        { "optimizer": 'ADAM', "lr": 0.001, "att_heads": 8 },
        { "optimizer": 'ADAM', "lr": 0.01, "att_heads": 8 },
        # { "optimizer": 'ADAM', "lr": 0.1, "att_heads": 8 },

        # { "optimizer": 'ADAM', "lr": 0.00001, "att_heads": 4 },
        # { "optimizer": 'ADAM', "lr": 0.0001, "att_heads": 4 },
        { "optimizer": 'ADAM', "lr": 0.001, "att_heads": 4 },
        { "optimizer": 'ADAM', "lr": 0.01, "att_heads": 4 },
        # { "optimizer": 'ADAM', "lr": 0.1, "att_heads": 4 },

        # { "optimizer": 'SGD', "lr": 0.0125, "att_heads": 8 },
        # { "optimizer": 'SGD', "lr": 0.025, "att_heads": 8 },
        { "optimizer": 'SGD', "lr": 0.05, "att_heads": 8 },
        { "optimizer": 'SGD', "lr": 0.1, "att_heads": 8 },
        # { "optimizer": 'SGD', "lr": 0.2, "att_heads": 8 },

        # { "optimizer": 'SGD', "lr": 0.0125, "att_heads": 4 },
        # { "optimizer": 'SGD', "lr": 0.025, "att_heads": 4 },
        { "optimizer": 'SGD', "lr": 0.05, "att_heads": 4 },
        { "optimizer": 'SGD', "lr": 0.1, "att_heads": 4 }
        # { "optimizer": 'SGD', "lr": 0.2, "att_heads": 4 }
        ]


    for p in hyperparams:
        params=Params(model=DGCNN, epochs=10, num_points=1024, emb_dims=1024, k=20, optimizer=p['optimizer'], lr=p['lr'], att_heads=p['att_heads'], momentum=0.9, dropout=0.5, dump_file=True, dry_run=False)
        train(params)
        test(params)

    for p in hyperparams:
        params=Params(model=AttentionDGCNN, epochs=10, num_points=1024, emb_dims=1024, k=20, optimizer=p['optimizer'], lr=p['lr'], att_heads=p['att_heads'], momentum=0.9, dropout=0.5, dump_file=True, dry_run=False)
        train(params)
        test(params)

    for p in hyperparams:
        params=Params(model=PointNet, epochs=20, num_points=1024, emb_dims=1024, k=20, optimizer=p['optimizer'], lr=p['lr'], att_heads=p['att_heads'], momentum=0.9, dropout=0.5, dump_file=True, dry_run=False)
        train(params)
        test(params)

    for p in hyperparams:
        params=Params(model=PointAttentionNet, epochs=20, num_points=1024, emb_dims=1024, k=20, optimizer=p['optimizer'], lr=p['lr'], att_heads=p['att_heads'], momentum=0.9, dropout=0.5, dump_file=True, dry_run=False)
        train(params)
        test(params)


    # if not args.eval:
         # train(args, io)
    # else:
    #     test(args, io)
