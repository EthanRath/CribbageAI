#import Cribbage
#import ChoosingModels
#import DataManager
#import MLFunctions
#import torch
import math

if __name__ == "__main__":
	pass
	#MLFunctions.Run_Training_Example()


a = [0]*1000
for i in range(1,1000):

	a[i] = math.cos(2*i) + a[i-1]
print(a)