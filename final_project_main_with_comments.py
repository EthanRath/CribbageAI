from Cribbage import *
from ChoosingModels import *
from MLFunctions import *
from DataManager import *
from matplotlib import pyplot as plt
import torch

### OHM: To account for macOS errors ###
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
### END EDITS ###

#This file mainly contains testing functions

#Trains the model on a subset of the data to show that the training code works while not taking too long to finish.
#I'm doing this instead of a doctest since doing a doctest with the training and testing functions isn't really viable.
#If you choose to run on the full dataset on the crib model with 10 epochs you will actually see the model over-fit eventually.
#cribmodel is a boolean denoting if you want to train a crib or a hand model
#fulldata is a boolean denoting if you wanna train on the whole dataset or a subset.
#learning_rate, bs, and epos are the training learning rate, batch size, and epochs respectively.
def DemoTraining(cribmodel = False, fulldata = False, learning_rate = 1e-3, bs = 128, epos = 5):

	#Crib model is given 2 cards and has to predict the value of the subsequent crib, while a hand model is given 4 cards and
	#has to predict the value. This boolean switches between the two if you want to test the crib model rather than the hand model.
	if not cribmodel:
	    model = NeuralNetwork([[20, 256], [256, 256], [256, 128], [128, 1]], True).to(device)
	else:
	    model = NeuralNetwork([[10, 256], [256, 256], [256, 128], [128, 1]], True).to(device)

	#Initialize the loss function and optimizer, here I am using Mean Square Error and Adam optimization
	loss_fn = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

	dataX = np.load("Data/dataX.npy")
	dataY = np.load("Data/dataY.npy")

	dataX = dataX[:10000*(1+9*fulldata)]
	dataY = dataY[:10000*(1+9*fulldata)] #shrinking data size for convenience if fulldata == False

	mult = 1
	if cribmodel:
	    dataX = np.concatenate((dataX[:, :10], dataX[:, 10:]), axis = 0) #crib model gets two inputs instead of 4
	    print(dataX.shape)
	    dataY = np.concatenate((dataY, dataY), axis = 0)
	    mult = 2

	#Split data into training and validation
	trainX = dataX[:9000*(1+9*fulldata)]
	testX = dataX[9000*(1+9*fulldata):]
	trainY = dataY[:9000*(1+9*fulldata)]
	testY = dataY[9000*(1+9*fulldata):]

	#Convert to tensors so pytorch doesn't get mad at me
	trainX = torch.Tensor(trainX)
	trainY = torch.Tensor(trainY)
	testX = torch.Tensor(testX)
	testY = torch.Tensor(testY)

	#Convert to data loaders so the model can train on the data
	trainData = DataLoader(Dataset_Custom(trainX, trainY), batch_size = bs)
	testData = DataLoader(Dataset_Custom(testX, testY), batch_size = bs)

	#Optimize the model
	trainloss, testloss = Optimize(trainData, testData, model, loss_fn, optimizer, epos)
	#trainloss = [elm.item() for elm in trainloss]
	#testloss = [elm.item() for elm in testloss]

	r = np.arange(0, epos, 1)
	plt.plot(r, trainloss)
	plt.xlabel("Epochs")
	plt.ylabel("Average Training Loss")
	plt.title("Model Training Performance Per Epoch")
	plt.show()

	plt.plot(r, testloss)
	plt.xlabel("Epochs")
	plt.ylabel("Average Testing Loss")
	plt.title("Model Testing Performance Per Epoch")
	plt.show()

	#Saves the model so you can use it later if you choose
	torch.save(model.state_dict(), "Models/Demo")

#Demonstrates the model's ability to predict the value of a hand before it knows the "cut card"
#On each printed line you will see the given hand, cut card, actual score, and model prediction.
#For predictions that are far off the actual value it is worth looking at the cut card to see
#what lucky (or unlucky) card was drawn to cause such a hard to predict outcome.
def DemoModelPerformance():
	x,y = GenerateData(100)
	enc = Encode(x)[:, :-5]

	model = NeuralNetwork([[20, 256], [256, 256], [256, 128], [128, 1]], True)
	model.load_state_dict(torch.load("Models/Attempt_3", map_location=torch.device('cpu')))
	model.eval()

	enc = torch.Tensor(enc)
	out = model(enc)

	ed = [", ", ", ", ", ","",""]
	for i in range(len(x)):
		for j in range(5):
			if j == 0:
				print("Hand: [", end = "")
			elif j == 4:
				print("] Cut: [", end = "")
			print(str(x[i][j]), end = ed[j])
		print("], ", end = "")
		print("Actual Value: " + str(y[i]), end = " | Model Prediction: ")
		print(float(out[i]))
		print()

#This function demonstrates the choosing capability of the "Chooser" model
#On each line you will see the 6 cards the model was delt followed by the 4 it chose to keep
#You can change the parameter "crib" to -1 to see how the choices differ when the opponent
#is going to receive the crib cards rather than the AI.
def DemoChoosingPerformance(crib = 1):
	#Load and initialize the hand and crib models
	model = NeuralNetwork([[20, 256], [256, 256], [256, 128], [128, 1]], True)
	model.load_state_dict(torch.load("Models/Attempt_3",  map_location=torch.device('cpu')))
	model.eval()

	model_c = NeuralNetwork([[10, 256], [256, 256], [256, 128], [128, 1]], True)
	model_c.load_state_dict(torch.load("Models/Crib_Attempt_1",  map_location=torch.device('cpu')))
	model_c.eval()

	#Create a hand chooser object using the hand and crib models
	Chooser = HandChooser(model, model_c, 1, .25)

	G = Game()
	ends = [", ", " | "]

	#Make iteratively generates hands and makes the chooser choose a hand
	#also prints out stuff so you know what is happening
	for i in range(100):
		hand, cut = G.FastDeal(n=6)
		choice = Chooser.Choose(hand, crib)
		print("Hand: ", end = "")
		for k in range(6):
		    print(str(hand[k]), end = ends[k==5])
		print("Choice: ", end = "")
		j = 0
		for k in range(6):
			if choice[k] == 0:
				j+=1
				print(hand[k], end = ends[j==4])
		print("\n", end = "\n")

#Uncomment the line(s) corresponding to whichever function(s) you want to run
if __name__ == "__main__":
    #DemoModelPerformance()
    #DemoChoosingPerformance()
    DemoTraining(True, True, epos = 100)
    #pass
    
    
### WWT Ethan,
###
###     This looks like a careful, informed implementation of Cribbage AI.
###     I don't know enough about cribbage to tell how well it is actually performing
###     from the point of view of a serious player of the game, but it looks
###     like it generally runs well and I could see that it learned (see detailed
###     comments below). Overall:  very nice work.
###
###     Issues (I think mostly minor, but maybe helpful to point them out):
###          ---I ran the doctesting and it mostly worked.  However, CountingHelpers.py
###             returned a doctest failure for ScorePairRunFlush
###          ---For some reason, whenever I revised main.py and reran DemoTraining, the first
###             time through, the model ran through whatever number of epochs I had specified
###             but it seemed to make no weight changes.  (The loss stayed constant).
###             However, when I simply reran the command "python main.py" from the terminal
###             the training proceeded normally the second time through.  (This was
###             on a mac).
###          ---I only tried training on the demo data.  I noticed that the model fairly
###             quickly overfit.  I ran it a couple of times to see around what point
###             it started overfitting (about 40 epochs) and then ran it again for that 
###             number of epochs.  It would be helpful if you made an option to check
###             for the test loss minimum and then install the weights at that point as
###             the final weight setting for a run.
###          ---I tried training it on 40 epochs and then, for comparison on 0 epochs. The
###             40 epoch version did decently and the 0 epoch seemed to do worse.  However, I got
###             the impression that even the 0 epoch version exhibited at least a weak
###             correlation between "Actual Value" and "Model Prediction". I noticed this
###             particularly with anomalously high Actual Values---namely those over 10:
###             although the model often gave small Predictions for small Actual Values,
###             it seemed to never do this  for the high Actual Values (all results over 6 and
###             many over 10).  This seems mysterious.  Of course I could be fooling myself
###             trying to peruse all those numbers.  Thus, it would be helpful if you ran
###             a regression model on Actual Value vs. Prediction and reported the signficance
###             of the correlation.  Is it possible there really is a positive correlation 
###             with no training?
###          ---Finally, I'm curious about two things:  (1) What is your assessment, as someone
###             who knows cribbage well, of the AI does when adequately trained?  (2) What, if
###             anything, has this project shown you about AI and/or cribbage?

### OHM: Good documentation and testing -- though I had to make some minor changes to ensure
### that the code ran without error on macOS. Not entirely your fault, but I do wonder if this could
### cause other downstream issues. Regardless all functions appear to work as you describe,
### though Whit raises some interesting questions.
###
### Overall, very good work.