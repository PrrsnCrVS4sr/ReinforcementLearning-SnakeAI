import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_size,out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size,out_features=output_size)
    def forward(self,x):
        return self.linear2((F.relu(self.linear1(x))))

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
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.opt = optim.Adam(self.model.parameters(),lr)
        self.loss = nn.MSELoss()
        
        pass

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state,dtype=torch.float)
        action = torch.tensor(action,dtype=torch.long)
        reward = torch.tensor(reward,dtype=torch.float)
        next_state = torch.tensor(next_state,dtype=torch.float)
        done = torch.tensor(done,dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state,0)
            next_state = torch.unsqueeze(next_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            done = (done,)
        #  each paramter can be single value or tupe to first convert to tensors
        # if single value, add a dimension using torch.unsqueeze
        Q_predicted = self.model(state)
        target = Q_predicted.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            if done[i] != True:
                Q_new += (torch.max(self.model(next_state[i])).item() * self.gamma)
            target[i][torch.argmax(action).item()] = Q_new
        # get predicted Q value with current state
        # 2. r + y* max(next_predicted Qvalue)
        self.opt.zero_grad()

        loss = self.loss(target,Q_predicted)

        loss.backward()

        self.opt.step()