import torch
import os
import numpy as np

def cosine_similarity(x1, x2):
    return np.dot(x1,x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def save_model(model, metric_func, optimizer, save_path, name, iteration):

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    filename = name+'_'+str(iteration)+'.pth'
    model_name = os.path.join(save_path, filename)
    torch.save({'model':model.state_dict(),
                'metric':metric_func.state_dict(),
                'optimizer':optimizer.state_dict()}, model_name)
    return model_name

def hitung_acc(outputs, labels):
    outputs = outputs.data.cpu().numpy()
    outputs = np.argmax(outputs, axis=1)
    labels = labels.data.cpu().numpy()
    acc = np.mean((outputs == labels).astype(int))
    return acc