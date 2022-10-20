import torch
import numpy as np 
import argparse

from configs import configuration
from dataloader import train_facedataloader
from models import Resnet
from margin import ArcFace
from test import model_evaluation
from utils import save_model, hitung_acc
from torch.utils.tensorboard import SummaryWriter
from margin import FocalLoss
from model import MobileFaceNet
if __name__ == "__main__":

    parser =  argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str, help='Experiment Name')

    args = parser.parse_args()
    writer = SummaryWriter("log/"+args.experiment_name)

    cfg = configuration()
    device = cfg.device

    # baca dataset dan dataloader
    print("Creating Images Dataset and Images Dataloader....", end="")
    train_dataset, train_dataloader = train_facedataloader()
    print("Done")

    # menentukan backbone
    if cfg.backbone == "resnet":
        print(f"Creating Model's Backbone : {cfg.backbone + str(50) + cfg.mode}....", end="")
        model = Resnet(embedding_size=cfg.embed_size, num_layers=cfg.num_layers, mode=cfg.mode)
        print("Done")   
    elif cfg.backbone == "mobilefacenet":
        print(f"Creating Model's Backbone : {cfg.backbone}....", end="")
        model = MobileFaceNet(embedding_size=cfg.embed_size)
        print("Done")

    # menentukan proses margin
    print("Creating Margin Function : ArcFace....", end="")
    margin_func = ArcFace(embed_size=cfg.embed_size, 
                          num_classes=cfg.num_classes, 
                          s=cfg.scale, m=cfg.margin)
    print("Done")

    if cfg.pretrained:
        print("Loading Model's state dict....", end="")
        states = torch.load(cfg.pretrained_path)
        model.load_state_dict(states['model'])
        margin_func.load_state_dict(states['metric'])
        print("Done")
    
    # transfer model to CUDA device and DataParallel module
    print(f"Transfering all Model's Parameters to {device} device....", end="")
    model.to(device)
    # transfer fungsi margin ke CUDA device and DataParallel module
    margin_func.to(device)
    print("Done")

    print(f"Creating Loss Function : {cfg.loss}....", end="")
    if cfg.loss == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif cfg.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    print("Done")

    print(f"Creating Optimizer : {cfg.opt}....", end="")
    # menentukan optimizer yang akan digunakan
    if cfg.opt == 'SGD':
        optimizer = torch.optim.SGD([{'params':model.parameters()}, 
                                     {'params':margin_func.parameters()}],
                                     lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.opt == 'Adam':
        optimizer = torch.optim.Adam([{'params':model.parameters()}, 
                                      {'params':margin_func.parameters()}], 
                                     lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.pretrained:
        states = torch.load(cfg.pretrained_path)
        optimizer.load_state_dict(states['optimizer'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step, gamma=0.1)
    print("Done")

    logfile = open(args.experiment_name + '.txt', 'w')
    print("Training Process Start ....")
    for i in range(cfg.epochs):
        train_acc_epoch = []
        train_loss = []
        model.train()
        for j, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            features = model(images)
            outputs = margin_func(features, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_acc = hitung_acc(outputs, labels)
            train_acc_epoch.append(train_acc)
            train_loss.append(loss.item())

            iteration = i * len(train_dataloader) + j

            writer.add_scalar('train accuracy', train_acc, iteration)
            writer.add_scalar('train loss', loss, iteration)

            if iteration % cfg.show_progress == 0:
                results = f"Epoch {i+1}/{cfg.epochs}, iteration : {iteration}, Train-Loss : {loss.item():.3f}, Train-Acc : {train_acc*100:.3f}%"
                print(results)
                logfile.write(results + '\n')

            if iteration % cfg.save_interval == 0:
                save_model(model, margin_func, 
                           optimizer, 
                           cfg.checkpoint_path, 
                           cfg.backbone+str(cfg.num_layers)+'_'+cfg.mode+'_'+str(train_acc)[:5], 
                           iteration)
            
        
        scheduler.step()
        results = f"Epoch {i+1}, Train-Acc : {np.mean(train_acc_epoch)*100:.3f}%, Train-Loss : {np.mean(train_loss):.3f}\n"
        print(results)
        logfile.write(results)
        model.eval()
        model_evaluation(model)

    writer.close()
    logfile.close()

