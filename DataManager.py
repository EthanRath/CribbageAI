import numpy as np
from Cribbage import Game
from Cribbage import Card
import torch
#from ChoosingModels import NeuralNetwork, HandChooser


def GenerateData(n):
	if n%2==1: #ensure n is even for convenience
		n += 1
	xdata = [0]*n
	ydata = [0]*n
	G = Game()
	for i in range(n):
		if i%1000 == 0:
			print("Iteration: ", i)
		hand, cut = G.FastDeal()
		ydata[i] = G.ScoreHand(hand, cut)
		xdata[i] = hand + [cut]

	return xdata, ydata

def Encode(hands, sum = -1):
	l = 5
	if sum != -1:
		l = 6
	encoding = np.zeros(shape = (len(hands), l*len(hands[0])))
	for i in range(len(hands)):
		for j in range(len(hands[i])):
			enc = hands[i][j].Encode()
			encoding[i, j*5:(j+1)*5] = enc
		if sum != -1:
			encoding[i, -1] = sum
	return encoding

def Encode_Play(playstack, hand):
    index = playstack[9]
    start = playstack[10]
    encoding = np.zeros(shape = (1, 12*6))
    for i in range(index):
        enc = playstack[i].Encode(True)
        enc[-1] = int(i <= start)
        encoding[0, i*6:(i+1)*6] = enc
    for i in range(len(hand)):
        enc = hand[i].Encode(True)
        enc[-1] = 1
        if i == 0:
            encoding[0, (-i-1)*6:] = enc
        else:
            encoding[0, (-i-1)*6:-i*6] = enc

    return encoding

if __name__ == "__main__":
	pass
	"""
	x,y = GenerateData(100)
	enc = Encode(x)

	testmodel = True
	if testmodel:
		model = NeuralNetwork([[20, 256], [256, 256], [256, 128], [128, 1]], True)
		model.load_state_dict(torch.load("Models/Attempt_3"))
		model.eval()

		enc = torch.Tensor(enc)
		out = model(enc)

	prnt = True
	if prnt:
		for i in range(len(x)):
			for j in range(5):
				ed = [", ", ": "]
				print(str(x[i][j]), end = ed[j == 4])
			print(y[i], end = " | Model Prediction: ")
			if testmodel:
				print(float(out[i]))
			print()

	save = False
	if save:
		temp = np.array(y)
		temp = temp[:, np.newaxis]
		print(temp.shape)
		np.save("Data/dataX5", enc)
		np.save("Data/dataY5", temp)
	"""
