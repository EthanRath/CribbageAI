import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from ChoosingModels import *
import numpy as np
import copy
from Cribbage import Game
from DataManager import Encode, Encode_Play
from matplotlib import pyplot as plt

#global variables aren't great but this is for convenience
#device = 'cpu' #to run on cpu
device = 'cuda' #to run on Nvidia GPU

class Dataset_Custom(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x, y):
        'Initialization'
        self.x = x
        self.y = y
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

  def __getitem__(self, index):
        return self.x[index], self.y[index]

def train_loop(dataloader, model, loss_fn, optimizer):
    optimizer.params = model.parameters()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def Optimize(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs = 10):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

def AdversarialTraining(hand, play, loss_fn, optimizer, batchsize = 1000, batches = 1000, eps = .1, decay = .1):
    g = Game()
    crib = 1
    chooser1, chooser2 = hand
    player1, player2 = play
    
    avghand1 = np.zeros(batches)
    avghand2 = np.zeros(batches)
    avgcrib1 = np.zeros(batches)
    avgcrib2 = np.zeros(batches)
    avgplay1 = np.zeros(batches)
    avgplay2 = np.zeros(batches)
    avgpeg1 = np.zeros(batches)
    avgpeg2 = np.zeros(batches)
    for i in range(batches):
        cribind = 0
        data_x = [np.zeros(shape = (batchsize, 20)), np.zeros(shape = (batchsize, 20))] #data for hand oracle of chooser
        data_y = [np.zeros(shape = (batchsize, 1)), np.zeros(shape = (batchsize, 1))]
        
        data_x_peg = [np.zeros(shape = (batchsize, 20)), np.zeros(shape = (batchsize, 20))] #data for play oracle of chooser
        data_y_peg = [np.zeros(shape = (batchsize, 1)), np.zeros(shape = (batchsize, 1))]
        
        data_x_crib = [np.zeros(shape = (batchsize//2, 10)), np.zeros(shape = (batchsize//2, 10))] #data for crib oracle of chooser
        data_y_crib = [np.zeros(shape = (batchsize//2, 1)), np.zeros(shape = (batchsize//2, 1))]
        
        data_x_play = [np.zeros(shape = (batchsize*4, 12*6)), np.zeros(shape = (batchsize*4, 12*6))] #data for play chooser
        data_y_play = [np.zeros(shape = (batchsize*4, 1)), np.zeros(shape = (batchsize*4, 1))]
        for j in range(batchsize):
            if j % 100 == 0:
                print("Iteration: ", j)
            h1, h2, cut = g.FastDeal_Dual()

            hand1, crib1 = chooser1.Choose(h1, crib, eps)
            hand2, crib2 = chooser2.Choose(h2, crib*-1, eps)
            handscore1 = g.ScoreHand(hand1, cut)
            handscore2 = g.ScoreHand(hand2, cut)
            
            data_x[0][j] = Encode([hand1])[0]
            data_x[1][j] = Encode([hand2])[0]
            data_x_peg[0][j] = Encode([hand1])[0]
            data_x_peg[1][j] = Encode([hand2])[0]
            data_y[0][j] = handscore1
            data_y[1][j] = handscore2
        
            cribscore = g.ScoreHand(crib1 + crib2, cut, crib=True)
            
            data_x_crib[crib == -1][cribind] = Encode([crib1])[0]
            data_y_crib[crib == -1][cribind] = cribscore
            cribind += j%2
            
            s1, s2 = SimPlayPhase(j, player1, player2, eps, hand1, hand2, g, crib, data_x_play[0], data_y_play[0], data_x_play[1], data_y_play[1], decay)
            
            data_y_peg[0][j] = s1
            data_y_peg[1][j] = s2
            crib = crib*-1
        print("Training Hand Choosers")
        train_hand1 = DataLoader(Dataset_Custom(data_x[0], data_y[0]), batch_size = batchsize)
        train_loop(train_hand1, chooser1.hand_model, loss_fn, optimizer)
        train_hand2 = DataLoader(Dataset_Custom(data_x[1], data_y[1]), batch_size = batchsize)
        train_loop(train_hand2, chooser2.hand_model, loss_fn, optimizer)
        avghand1[i] = np.mean(data_y[0])
        avghand2[i] = np.mean(data_y[1])
        print("Average Hand 1 Score Batch " + str(i) + ": " + str(avghand1[i]))
        print("Average Hand 2 Score Batch " + str(i) + ": " + str(avghand2[i]))
        
        print("Training Crib Choosers")
        train_crib1 = DataLoader(Dataset_Custom(data_x_crib[0], data_y_crib[0]), batch_size = batchsize)
        train_loop(train_crib1, chooser1.crib_model, loss_fn, optimizer)
        train_crib2 = DataLoader(Dataset_Custom(data_x_crib[1], data_y_crib[1]), batch_size = batchsize) 
        train_loop(train_crib2, chooser2.crib_model, loss_fn, optimizer)
        avgcrib1[i] = np.mean(data_y_crib[0])
        avgcrib2[i] = np.mean(data_y_crib[1])
        print("Average Crib 1 Score Batch " + str(i) + ": " + str(avgcrib1[i]))
        print("Average Crib 2 Score Batch " + str(i) + ": " + str(avgcrib2[i]))
        
        print("Training Players")
        train_play1 = DataLoader(Dataset_Custom(data_x_play[0], data_y_play[0]), batch_size = batchsize) 
        train_loop(train_play1, player1.model, loss_fn, optimizer)
        train_play2 = DataLoader(Dataset_Custom(data_x_play[1], data_y_play[1]), batch_size = batchsize) 
        train_loop(train_play2, player2.model, loss_fn, optimizer)
        avgplay1[i] = np.mean(data_y_play[0])
        avgplay2[i] = np.mean(data_y_play[1])
        print("Average Play 1 Score Batch " + str(i) + ": " + str(avgplay1[i]))
        print("Average Play 2 Score Batch " + str(i) + ": " + str(avgplay2[i]))
        
        print("Training Play Choosers")
        train_peg1 = DataLoader(Dataset_Custom(data_x_peg[0], data_y_peg[0]), batch_size = batchsize) 
        train_loop(train_peg1, chooser1.play_model, loss_fn, optimizer)
        train_peg2 = DataLoader(Dataset_Custom(data_x_peg[1], data_y_peg[1]), batch_size = batchsize) 
        train_loop(train_peg2, chooser2.play_model, loss_fn, optimizer)
        avgpeg1[i] = np.mean(data_y_peg[0])
        avgpeg2[i] = np.mean(data_y_peg[1])
        print("Average Peg 1 Score Batch " + str(i) + ": " + str(avgpeg1[i]))
        print("Average Peg 2 Score Batch " + str(i) + ": " + str(avgpeg2[i]))
        
    x = np.arange(0,batches, 1)
    
    plt.figure(dpi = 500)
    plt.plot(x, avghand1)
    plt.plot(x, avghand2)
    plt.title("Avg hand score over batches")
    plt.show()
    
    plt.figure(dpi = 500)
    plt.plot(x, avgcrib1)
    plt.plot(x, avgcrib2)
    plt.title("Avg crib score over batches")
    plt.show()
    
    plt.figure(dpi = 500)
    plt.plot(x, avgplay1)
    plt.plot(x, avgplay2)
    plt.title("Avg play score over batches")
    plt.show()
    
    plt.figure(dpi = 500)
    plt.plot(x, avgpeg1)
    plt.plot(x, avgpeg2)
    plt.title("Avg peg score over batches")
    plt.show()

  

def SimPlayPhase(j, p1, p2, eps, hand1, hand2, g, crib, dx1, dy1, dx2, dy2, decay, prnt = False, train = True):
    
    th1 = copy.copy(hand1)
    th2 = copy.copy(hand2)
    gos = 0
    ind1 = 0
    ind2 = 0
    s1 = 0
    s2 = 0
    while True:
        if len(th1) == 0 and len(th2) == 0:
            break
        if gos >= 2:
            gos = 0
            g.Go()
            if train and crib == -1:
                dy1[(4*j)+ind1 - 1] += 1
            if train and crib == 1:
                dy2[(4*j)+ind2 - 1] += 1
            if prnt:
                print("Go for 1 point")
                print("------------------------")
            continue
        if crib == -1 or len(th2) == 0:
            card, ind, cango = p1.Play(g.playstack, th1, g.playstack[8], eps)
            if cango:
                score = g.Play_Phase(card)
                if prnt:
                    print("Player 1 Hand: ", [str(c) for c in th1])
                    print("Current Stack: ", [str(g.playstack[i]) for i in range(g.playstack[10], len(g.playstack))])
                    print("Card Chosen: ", str(card))
                    print("Score: ", score)
                    print()
                th1 = th1[:ind] + th1[ind+1:]
                crib = 1
                if train:
                    dy1[(4*j) + ind1] = score
                    if ind1 > 0: #reward self
                        dy1[(4*j) + ind1 - 1] += (score*decay)
                    if ind2 > 0: #punish other
                        dy2[(4*j) + ind2-1] -= (decay*score)
                    dx1[(4*j) + ind1] = Encode_Play(g.playstack, th1)[0]
                ind1 += 1
            else:
                gos += 1
        elif crib == 1 or len(th1) == 0:
            card, ind, cango = p2.Play(g.playstack, th2, g.playstack[8], eps)
            if cango:
                score = g.Play_Phase(card)
                if prnt:
                    print("Player 2 Hand: ", [str(c) for c in th2])
                    print("Current Stack: ", [str(g.playstack[i]) for i in range(g.playstack[10], len(g.playstack))])
                    print("Card Chosen: ", str(card))
                    print("Score: ", score)
                    print()
                th2 = th2[:ind] + th2[ind+1:]
                crib = -1
                if train:
                    dy2[(4*j) + ind2] = score
                    if ind2 > 0: #reward self
                        dy2[(4*j) + ind2-1] += (decay*score)
                    if ind1 > 0: #punish other
                        dy1[(4*j) + ind1 - 1] -= (score*decay)
                    dx2[(4*j) + ind2] = Encode_Play(g.playstack, th2)[0]
                ind2 += 1
            else:
                gos += 1
    g.ResetPlay()
    if train:
        s1 = np.sum(dy1[(j*4):(j*4) + 4])
        s2 = np.sum(dy2[(j*4):(j*4) + 4])
        return s1,s2
        

def Run_Training_Example():
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #loss_fn = nn.CrossEntropyLoss()

    cribmodel = True
    if not cribmodel:
        model = NeuralNetwork([[20, 256], [256, 256], [256, 128], [128, 1]], True).to(device)
    else:
        model = NeuralNetwork([[10, 256], [256, 256], [256, 128], [128, 1]], True).to(device)
    #model = NeuralNetwork().to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    dataX = np.load("Data/dataX.npy")
    dataY = np.load("Data/dataY.npy")

    dataX = dataX[:10000]
    dataY = dataY[:10000] #shrinking data size for convenience

    mult = 1
    if cribmodel:
        dataX = np.concatenate((dataX[:, :10], dataX[:, 10:]), axis = 0) #crib model gets two inputs instead of 4
        print(dataX.shape)
        dataY = np.concatenate((dataY, dataY), axis = 0)
        mult = 2

    trainX = dataX[:9000]
    testX = dataX[9000:]
    trainY = dataY[:9000]
    testY = dataY[9000:]

    trainX = torch.Tensor(trainX)
    trainY = torch.Tensor(trainY)
    testX = torch.Tensor(testX)
    testY = torch.Tensor(testY)

    trainData = DataLoader(Dataset_Custom(trainX, trainY), batch_size = 128)
    testData = DataLoader(Dataset_Custom(testX, testY), batch_size = 128)

    Optimize(trainData, testData, model, loss_fn, optimizer)

    torch.save(model.state_dict(), "Models\\Demo")
    
def Run_Adv_Training():
    chooser1 = NeuralNetwork([[20, 256], [256, 256], [256, 128], [128, 1]], True).to(device).double()
    crib1 = NeuralNetwork([[10, 256], [256, 256], [256, 128], [128, 1]], True).to(device).double()
    peg1 = NeuralNetwork([[20, 256], [256, 256], [256, 128], [128, 1]], True).to(device).double()
    play1 = NeuralNetwork([[6*12, 256], [256, 256], [256, 128], [128, 1]], True).to(device).double()
    
    chooser2 = NeuralNetwork([[20, 256], [256, 256], [256, 128], [128, 1]], True).to(device).double()
    crib2 = NeuralNetwork([[10, 256], [256, 256], [256, 128], [128, 1]], True).to(device).double()
    peg2 = NeuralNetwork([[20, 256], [256, 256], [256, 128], [128, 1]], True).to(device).double()
    play2 = NeuralNetwork([[6*12, 256], [256, 256], [256, 128], [128, 1]], True).to(device).double()

    choose1 = HandChooser(chooser1, crib1, peg1, 1, .15, .05)
    choose2 = HandChooser(chooser2, crib2, peg2, 1, .15, .05)
    
    player1 = AI_Player(play1)
    player2 = AI_Player(play2)
    
    learning_rate = 1e-3
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(chooser1.parameters(), lr = learning_rate)
    AdversarialTraining([choose1, choose2], [player1, player2], loss_fn, optimizer, 100, 100)
    

    choose1.save("Saves/C1")
    choose2.save("Saves/C2")
    player1.save("Saves/P1")
    player2.save("Saves/P2")
    
def Load_And_Sim():
    chooser1 = NeuralNetwork([[20, 256], [256, 256], [256, 128], [128, 1]], True).to(device).double()
    crib1 = NeuralNetwork([[10, 256], [256, 256], [256, 128], [128, 1]], True).to(device).double()
    peg1 = NeuralNetwork([[20, 256], [256, 256], [256, 128], [128, 1]], True).to(device).double()
    play1 = NeuralNetwork([[6*12, 256], [256, 256], [256, 128], [128, 1]], True).to(device).double()
    
    chooser2 = NeuralNetwork([[20, 256], [256, 256], [256, 128], [128, 1]], True).to(device).double()
    crib2 = NeuralNetwork([[10, 256], [256, 256], [256, 128], [128, 1]], True).to(device).double()
    peg2 = NeuralNetwork([[20, 256], [256, 256], [256, 128], [128, 1]], True).to(device).double()
    play2 = NeuralNetwork([[6*12, 256], [256, 256], [256, 128], [128, 1]], True).to(device).double()

    choose1 = HandChooser(chooser1, crib1, peg1, 1, .15, .05)
    choose2 = HandChooser(chooser2, crib2, peg2, 1, .15, .05)
    
    player1 = AI_Player(play1)
    player2 = AI_Player(play2)
    
    learning_rate = 1e-3
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(chooser1.parameters(), lr = learning_rate)
    #AdversarialTraining([choose1, choose2], [player1, player2], loss_fn, optimizer, 1000, 1000)
    

    choose1.load("Saves/C1")
    choose2.load("Saves/C2")
    player1.load("Saves/P1")
    player2.load("Saves/P2")
    
    g = Game()
    crib = 1
    for i in range(2):
        h1, h2, cut = g.FastDeal_Dual()
        hand1, crib1 = choose1.Choose(h1, crib, 0)
        hand2, crib2 = choose2.Choose(h2, crib*-1, 0)
        
        print("Player 1 delt: ", [str(c) for c in h1])
        print("Player 1 Chose: ", [str(c) for c in hand1])
        print("-----------------------------------")
        print("Player 2 delt: ", [str(c) for c in h2])
        print("Player 2 Chose: ", [str(c) for c in hand2])
        print("-----------------------------------")
        
        SimPlayPhase(i, player1, player2, 0, hand1, hand2, g, crib, None, None, None, None, 0, prnt = True, train = False)
    


if __name__ == "__main__":
    Load_And_Sim()
    #Run_Adv_Training()
