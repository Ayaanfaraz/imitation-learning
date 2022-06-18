from msilib.schema import File
from types import AsyncGeneratorType
import numpy as np
import matplotlib.pyplot
import matplotlib.pyplot as plt
from mergedModel import MyEnsemble as fusionModel
import matplotlib.pyplot as plt
import xception
import torch.nn as nn
import torch.optim as optim
import torch
import pickle

import tiramisuModel.tiramisu as tiramisu
from torchvision import transforms
from mergedModel import MyEnsemble as fusionModel

#Variables
MEMORY_FRACTION = 0.6
WIDTH = 300
HEIGHT = 300
EPOCHS=10
MODEL_NAME = "Xception"
TRAINING_BATCH_SIZE = 32

TIMESTEPS_PER_EPISODE = 40
EPISODES = 5

class DQNAgent:
    def __init__(self):
       
        self.model = fusionModel(semantic_model=self.create_model(), uncertainty_model=self.create_model())#.to(device='cuda:1')
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "model" in name:
                    param.requires_grad = False
                else:
                    print(name) 

        #self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)

    def create_model(self):
        return xception.xception(num_classes=2048, pretrained=False)#.to(device='cuda:1')

    def train(self):

        torch.cuda.empty_cache()
        states, supervised_labels = [],[]
        torch.autograd.set_detect_anomaly(True)
        criterion = nn.MSELoss()#.to(device='cuda:1')
        uncertainty_states = []
        semantic_states = []

        actions_list = []
        images_list = []
        
        with open('_out/data.pkl','rb') as af:
            actions_list = pickle.load(af)
        
        with open('_out/images.pkl','rb') as f:
            images_list = pickle.load(f)

        for i in range(len(images_list)): #Change this to reading in input.

            semantic_state = (torch.from_numpy(images_list[i][0]).permute(2,0,1)/255)#.to(device='cuda:1')
            uncertainty_state = (torch.from_numpy(images_list[i][1]).permute(2,0,1)/255)#.to(device='cuda:1')

            semantic_states.append(semantic_state) #Add the uncertainty/semantic segmented tuple
            uncertainty_states.append(uncertainty_state)

            del semantic_state
            del uncertainty_state

            supervised_labels.append(torch.tensor(actions_list[i])) # 16 1 by 3 tensors (list of q value outputs
        
        
        self.model.train()#.to(device='cuda:1')
        #del self.loss
        for i in range(len(semantic_states)):
            self.optimizer.zero_grad()
            x1    = torch.stack((semantic_states[i:i+int(TRAINING_BATCH_SIZE)]))#Semantic -> [[3x4]] -> 1x3x4 -> #
            x2    = torch.stack((uncertainty_states[i:i+int(TRAINING_BATCH_SIZE)]))  #states[i][1]#Uncertainty

            y     = torch.stack(supervised_labels[i:i+int(TRAINING_BATCH_SIZE)]) #batch size of 4 labels
            # print("ys shape: ", y.shape)
            # print("y shape is:", y.shape)
            yhat = self.model(x1,x2)
            # print("yhat shape: ", yhat.shape)
            # print("yhat is: ",yhat.shape) #4,3 tensor
            
            loss=criterion(yhat, y)
            loss.backward()
            self.optimizer.step()

            i+=TRAINING_BATCH_SIZE
            print("Loss: ", loss)

agent = DQNAgent()
agent.train()

