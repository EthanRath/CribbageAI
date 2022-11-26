import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import OrderedDict
from DataManager import Encode, Encode_Play
from Cribbage import Game
import random
import copy

class NeuralNetwork(nn.Module):
    def __init__(self, dims = [], od = False):
        super(NeuralNetwork, self).__init__()

        if od:
            temp = OrderedDict()
            names = ["Linear", "Relu"]
            for i in range(len(dims)):
                dim = dims[i]
                temp[names[0] + str(i)] = nn.Linear(dim[0], dim[1])
                temp[names[1] + str(i)] = nn.ReLU()
            self.linear_relu_stack = nn.Sequential(temp)

        else:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(20, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.ReLU()
            )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class HandChooser:

    def __init__(self, hand_model, crib_model, play_model, hand_bias, crib_bias, play_bias, device = 'cuda'):
        self.hand_model = hand_model
        self.crib_model = crib_model
        self.play_model = play_model
        self.play_bias = play_bias
        self.hand_bias = hand_bias
        self.crib_bias = crib_bias
        self.device = device

    def save(self,path):
        torch.save(self.hand_model.state_dict(), path + "\\chooser")
        torch.save(self.crib_model.state_dict(), path + "\\crib")
        torch.save(self.play_model.state_dict(), path + "\\peg")
        
    def load(self,path):
        self.hand_model.load_state_dict(torch.load(path + "\\chooser"))
        self.crib_model.load_state_dict(torch.load(path + "\\crib"))
        self.play_model.load_state_dict(torch.load(path + "\\peg"))
        self.hand_model.eval()
        self.crib_model.eval()
        self.play_model.eval()

    #crib = 1 if player crib, -1 if opponent crib
    def Choose(self, hand, crib = 1, eps = 0):
        scores = [0]*15
        subs = [0]*15
        o1 = 0
        o2 = 1
        hand_t = [0]*4 #holds hand to be scored
        crib_t = [0]*2 #holds crib to be scored
        max_val = None
        max_choice = None
        for i in range(15):
            co = 0
            ho = 0
            if i!=0 and (i+o2)%6 == 0:
                o1 += 1
                o2 += o1+1
            sub = [0 for iterator in range(6)]
            sub[o1] = 1
            sub[(i+o2)%6] = 1
            for j in range(len(sub)):
                if sub[j] == 1:
                    crib_t[co] = hand[j]
                    co+=1
                else:
                    hand_t[ho] = hand[j]
                    ho += 1
            hand_e = torch.Tensor(Encode([hand_t])).to(self.device).double()
            crib_e = torch.Tensor(Encode([crib_t])).to(self.device).double()
            temp = (self.hand_bias * float(self.hand_model(hand_e)) +
                    (self.crib_bias * float(self.crib_model(crib_e)) * crib)
                    + (self.play_bias * self.play_model(hand_e))) #score for this choice of hand
            scores[i] = temp
            subs[i] = sub
            if max_val == None or max_val < temp:
                max_choice = sub
                max_val = temp
        if random.random() < eps:
            max_choice = subs[random.randint(0, 14)]

        res = [[],[]]
        for i in range(len(hand)):
            if max_choice[i] == 0:
                res[0].append(hand[i])
            else:
                res[1].append(hand[i])
        return res[0], res[1], scores

class AI_Player:
    def __init__(self, model, device = 'cuda'):
        self.model = model
        self.device = device

    def load(self,path):
        self.model.load_state_dict(torch.load(path + "\\play"))
        self.model.eval()

    def save(self, path):
        torch.save(self.model.state_dict(), path + "\\play")

    #state is the play stack
    def Play(self, state, hand, sum, eps = 0):
        scores = [0]*len(hand)
        ind = 0
        cango = False
        for i in range(len(hand)):
            temp_state = copy.copy(state)
            if sum + hand[i].value > 31:
                scores[i] = -100
                continue
            cango = True
            temp_state[temp_state[9]] = hand[i] 
            temp = torch.tensor(Encode_Play(temp_state, hand[:i] + hand[i+1:])).to(self.device)
            scores[i] = self.model(temp)
            if i>0 and scores[i] > scores[ind]:
                ind = i
        if random.random() > eps:
            while True:
                ind = random.randint(0, len(hand) - 1)
                if hand[ind] != -100:
                    break
        return hand[ind], ind, cango


if __name__ == "__main__":
    model = NeuralNetwork([[20, 256], [256, 256], [256, 128], [128, 1]], True)
    model.load_state_dict(torch.load("Models/Attempt_3"))
    model.eval()

    model_c = NeuralNetwork([[10, 256], [256, 256], [256, 128], [128, 1]], True)
    model_c.load_state_dict(torch.load("Models/Crib_Attempt_1"))
    model_c.eval()

    Chooser = HandChooser(model, model_c, 1, .25)

    G = Game()
    ends = [", ", " | "]

    for i in range(50):
        G.Shuffle()
        G.DealCards()
        for j in range(2):
            hand = G.hands[j]
            choice = Chooser.Choose(hand)
            for k in range(6):
                print(str(hand[k]), end = ends[k==5])
            print("Choice: ", end = "")
            for k in range(6):
                if choice[k] == 0:
                    print(hand[k], end = ends[k==5])
            print("\n", end = "\n")
