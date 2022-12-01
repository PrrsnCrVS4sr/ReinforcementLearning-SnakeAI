import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        pass
    def forward(self,x):
        pass

    def save(self,file_name ='model.pth'):
        model_folder_path = './model'

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self,model,lr,gamma):
        #set stuff like optimisers
        #use MSE loss
        pass

    def train_step(self, state, action, reward, next_state, done):
        #  each paramter can be single value or tupe to first convert to tensors
        # if single value, add a dimension using torch.unsqueeze

        # get predicted Q value with current state
        # 2. r + y* max(next_predicted Qvalue)
        pass