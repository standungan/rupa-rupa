import os
import cv2
import torch
import numpy as np
from configs import configuration
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from configs import configuration
from models import Resnet 

def cosine_similarity(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1)*np.linalg.norm(x2))

def calculate_similarity(model):
    cfg = configuration()
    
    transformer = transforms.Compose([
        transforms.Resize(cfg.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], 
                                std=[0.5,0.5,0.5])
    ])

    with open(cfg.test_list, 'r') as handler:
        pairs = handler.readlines()

    similarities = []
    labels = []
    previous_src = " "

    for _, pair in enumerate(tqdm(pairs)):
        splits = pair.split()
        src_name = cfg.source_root + splits[0]
        tgt_name = cfg.target_root + splits[1]
        label = int(splits[2])

        if previous_src != src_name:
            image_src = transformer(Image.open(src_name)).unsqueeze(0)
            embed_src = model(image_src.to(cfg.device)).cpu().detach().numpy()
            previous_src = src_name
    
        image_tgt = transformer(Image.open(tgt_name)).unsqueeze(0)
        embed_tgt = model(image_tgt.to(cfg.device)).cpu().detach().numpy() 

        cos_sim = cosine_similarity(embed_src, np.transpose(embed_tgt, (1,0)))

        similarities.append(cos_sim.item())
        labels.append(label)

    return similarities, labels

def calculate_accuracy(similarities, labels):
        
    similarities = np.asarray(similarities)
    labels = np.asarray(labels)
    best_acc = 0
    best_tpr = 0
    best_fpr = 100
    best_th = 0

    for i in tqdm(range(len(similarities))):
        threshold = similarities[i]
        y_test = (similarities >= threshold)
        accuracy = np.mean((y_test==labels).astype(int))
        tpr = calculate_tpr(y_test, labels)
        fpr = calculate_fpr(y_test, labels)
        if accuracy > best_acc:
            best_fpr = fpr
            best_tpr = tpr
            best_th = threshold
            best_acc = accuracy

    return (best_acc, best_th, best_tpr, best_fpr)

def calculate_tpr(y_test, labels):
    P = np.where(labels == 1)[0]
    return np.sum(y_test[P]) * 100 / len(P)

def calculate_fpr(y_test, labels):
    N = np.where(labels == 0)[0]
    return np.sum(y_test[N]) * 100 / len(N)

def model_evaluation(model):

    print("Calculating Similarity between faces....")
    similarities, labels  = calculate_similarity(model)

    print("Calculating Model' accuracy....")
    accuracy, threshold, tpr, fpr = calculate_accuracy(similarities, labels)

    print("Testing Accuracy : {:.3f}%, Best Threshold : {:.3f}".format(accuracy*100, threshold))
    print("Overall True Positive Rate  : {:.3f}%".format(tpr))
    print("Overall False Positive Rate : {:.3f}%".format(fpr))

    # return (accuracy, threshold, tpr, fpr)

if __name__ == "__main__":
    
    print("Load configuration file....Done")
    cfg = configuration()

    print("Create model and load model's training state....", end="")
    model = Resnet(embedding_size=cfg.embed_size, 
                   num_layers=cfg.num_layers, 
                   mode=cfg.mode)
    
    states = torch.load('checkpoint/ResNet50_ir_se_0.0_1000.pth')
    model.load_state_dict(states['model'])
    model.to(cfg.device)

    # states = torch.load("weights/resnet18_110.pth")
    # model.load_state_dict(states)

    model.eval()
    model_evaluation(model)    
