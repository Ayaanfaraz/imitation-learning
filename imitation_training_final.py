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
from tqdm import tqdm
import random
from sklearn.metrics import mean_squared_error, r2_score
#import sklearn.model_selection import train_test_split
import tiramisuModel.tiramisu as tiramisu
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from mergedModel import MyEnsemble as fusionModel

# command to run the logs on the Tensorboar:
# tensorboard dev upload --logdir="C:\Users\s1929247\Documents\Ali-Document\Computer Science\Project\imitation\runs\Jun20_14-21-17_AP4SBLJG3"

#Variables
WIDTH = 300
HEIGHT = 300
EPOCHS = 300

TOTAL_LENGTH =  68932 #53486 #68932
TRAINING_BATCH_SIZE = 128 #64 #32, 128 #256
TRAIN_SIZE =int(TOTAL_LENGTH* 0.80)
LRATE = 0.0005

device = torch.device("cuda:2")
device2 = torch.device("cuda:2")
# device = torch.device("cpu")
torch.cuda.empty_cache() 

class imitation:
    def __init__(self):

        self.model = fusionModel(semantic_model=self.create_model(), uncertainty_model=self.create_model())#.to(device)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "model" in name:
                    param.requires_grad = False
                #else:
                   # print(name)

        #self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)

    def create_model(self):
        return xception.xception(num_classes=2048, pretrained=False)#.to(device)

    def getStatesandLabels(self, batch_size, af, f):
        semantic_states, uncertainty_states, supervised_labels = [],[],[]
        for i in range(batch_size):
            try:
                action_tuple = pickle.load(af)
                images_tuple = pickle.load(f)
                semantic_state = (torch.from_numpy(images_tuple[0]).permute(2,0,1)/255)#.to(device)
                uncertainty_state = (torch.from_numpy(images_tuple[1]).permute(2,0,1)/255)#.to(device)

                semantic_states.append(semantic_state) #Add the uncertainty/semantic segmented tuple
                uncertainty_states.append(uncertainty_state)

                supervised_labels.append(torch.tensor(action_tuple)) #16 1 by 3 tensors (list of q value outputs
            except:
                break 

        return semantic_states, uncertainty_states, supervised_labels

    def train(self):

        torch.cuda.empty_cache()
        torch.autograd.set_detect_anomaly(True)
        criterion = nn.MSELoss()#.to(device)
        
        self.model.train().to(device)
        tb = SummaryWriter()
        tb = SummaryWriter()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=LRATE)
        self.model.train().to(device)

        for epoch in range(EPOCHS):
            action_file = open('_out/imitation_training_data_optimized.pkl','rb')
            states_file = open('_out/imitation_training_images_optimized.pkl','rb')
            for i in range(0,TRAIN_SIZE,TRAINING_BATCH_SIZE):

                semantic_states, uncertainty_states,supervised_labels\
                    = self.getStatesandLabels(batch_size=TRAINING_BATCH_SIZE, af=action_file, f=states_file)

                epoch_itter = epoch * TRAIN_SIZE + i
                self.optimizer.zero_grad()
                x1    = torch.stack(semantic_states)#Semantic -> [[3x4]] -> 1x3x4 -> #
                x2    = torch.stack(uncertainty_states)  #states[i][1]#Uncertainty
                y     = torch.stack(supervised_labels) #batch size of 4 labels
                y = y.type(torch.FloatTensor)

                x1, x2, y  = x1.to(device), x2.to(device), y.to(device)

                yhat = self.model(x1,x2)
                #print("y shape:", y.shape, "yhat shape: ",yhat.shape)
                #print("Y type: ", y.dtype, "yhat type: ", yhat.dtype)


                loss=criterion(yhat, y)
                loss.backward()
                self.optimizer.step()

                del x1
                del x2
                torch.cuda.empty_cache()
                print(f"Loss: lrate-epoch-itter: {LRATE}-{epoch}-{i}", loss.item())

                steer_accuracy = throttle_accuracy = brake_accuracy = 0

                #print("YHAT : ", round(yhat[0,0].item(), 2))
                #print("Y : ", torch.round(y[0,2], decimals=2))
                for j in range(len(y)):

                    steer_accuracy += 1-((abs(round(y[j,0].item(), 2) - round(yhat[j,0].item(), 2))+1)/(round(y[j,0].item(), 2)+1))
                    throttle_accuracy += 1-((abs(round(y[j,1].item(), 2) - round(yhat[j,1].item(), 2))+1)/(round(y[j,1].item(), 2)+1))
                    brake_accuracy += 1-((abs(round(y[j,2].item(), 2) - round(yhat[j,2].item(), 2))+1)/(round(y[j,1].item(), 2)+1))

                accuracy_steer = round(steer_accuracy/len(y), 2)
                accuracy_throttle = round(throttle_accuracy/len(y), 2)
                accuracy_brake = round(brake_accuracy/len(y), 2)
                accuracy_avg = round((accuracy_steer + accuracy_throttle + accuracy_brake) / 3, 2)

                tb.add_scalar("Loss", loss, epoch_itter)

                tb.add_scalar("Steer Accuracy", accuracy_steer, epoch_itter)
                tb.add_scalar("Throttle Accuracy", accuracy_throttle, epoch_itter)
                tb.add_scalar("Brake Accuracy", accuracy_brake, epoch_itter)
                tb.add_scalar("Accuracy Average", accuracy_avg, epoch_itter)

                # Makes sure we dont consider out of bounds scenarios
                if i + TRAINING_BATCH_SIZE > TRAIN_SIZE:
                    break

######Validating the model
            print("Training complete, validating model")
            torch.cuda.empty_cache()
            with torch.no_grad():
                for i in range(TRAIN_SIZE,TOTAL_LENGTH):
                    self.model.eval().to(device2)
                    semantic_states, uncertainty_states,supervised_labels = self.getStatesandLabels(batch_size=TRAINING_BATCH_SIZE, af=action_file, f=states_file)
                    if len(supervised_labels) < TRAINING_BATCH_SIZE:
                        break
                    x1_test    = torch.stack(semantic_states)#Semantic -> [[3x4]] -> 1x3x4 -> #
                    x2_test    = torch.stack(uncertainty_states)  #states[i][1]#Uncertainty
                    y_test     = torch.stack(supervised_labels) #batch size of 4 labels

                    x1_test, x2_test, y_test  = x1_test.to(device2), x2_test.to(device2), y_test.to(device2)
                    #print("x1 test shape: ",x1_test.shape)
                    yhat_test = self.model(x1_test, x2_test)

                    del x1_test
                    del x2_test
                    torch.cuda.empty_cache()

                    test_correct_steer = test_correct_throttle = test_correct_brake = 0
                    for j in range(len(y_test)):
                        test_correct_steer += 1-((abs(round(y[j,0].item(), 2) - round(yhat[j,0].item(), 2))+1)/(round(y[j,0].item(), 2)+1))
                        test_correct_throttle += 1-((abs(round(y[j,1].item(), 2)+1 - round(yhat[j,1].item(), 2))+1)/(round(y[j,1].item(), 2)+1))
                        test_correct_brake += 1-((abs(round(y[j,2].item(), 2)+1 - round(yhat[j,2].item(), 2))+1)/(round(y[j,1].item(), 2)+1))

                    test_accuracy_steer = round(test_correct_steer/len(y_test), 2)
                    test_accuracy_throttle = round(test_correct_throttle/len(y_test), 2)
                    test_accuracy_brake = round(test_correct_brake/len(y_test), 2)
                    test_accuracy_avg = round((test_accuracy_steer + test_accuracy_throttle + test_accuracy_brake) / 3, 2)

                    y_test = y_test.cpu().numpy()
                    yhat_test = yhat_test.cpu().numpy()

                    tb.add_scalar("Validation Accuracy Average", test_accuracy_avg, epoch)
                    tb.add_scalar("Validation Accuracy Steer", test_accuracy_steer, epoch)
                    tb.add_scalar("Validation Accuracy Throttle", test_accuracy_throttle, epoch)
                    tb.add_scalar("Validation Accuracy Brake", test_accuracy_brake, epoch)

                    #tb.add_scalar("validation_R2", r_square, epoch)

                    del y_test
                    del yhat_test
                    #del mse
                    # del r_square
                    torch.cuda.empty_cache()
                    print("Validating: ", i)
                    if (i + TRAINING_BATCH_SIZE) >= TOTAL_LENGTH:
                        break
            action_file.close()
            states_file.close()
        tb.close()
        print("TRAINING FINISHED")
        torch.save(self.model,f'imitation_models/model_final_lr_{lrate}.pt')
        torch.cuda.empty_cache()

agent = imitation()
agent.train()
