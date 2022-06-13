import math
import random
import numpy as np
import copy
from ScoringUtils import PlayPairs, PlayRun


class Card:
	def __init__(self, suit, value, order):
		self.suit = suit
		self.value = value
		self.order = order
		self.suit_enc = ['heart', 'spade', 'clover', 'diamond'].index(self.suit)
		self.Null = False
	def __str__(self):
		cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
		return cards[self.order - 1] + " of " + self.suit

	def Encode(self, play = False):
		encoding = np.zeros(5 + play)
		if self.Null:
			return encoding
		encoding[self.suit_enc] = 1
		encoding[4] = self.order
		return encoding


class Game:
	def __init__(self):
		self.deck = self.GenerateDeck()
		self.score = [0,0]
		self.hands = [[0]*6,[0]*6]
		self.deckPos = 0
		self.playstack = [0]*11 #8th position contains stack score, 9th contains stack index, 10th contains start index of current round

	def GenerateDeck(self):
		suits = ['heart', 'spade', 'clover', 'diamond']
		deck = [0]*52
		#Creates the deck O(n)
		for j in range(len(suits)):
			for i in range(1,14):
				suit = suits[j]
				pos = j*13 + i - 1
				value = min(i, 10)
				deck[pos] = Card(suit, value, i)
		return deck

	#O(n) shuffle algorithm
	#Here I am using systems like shuffle and deal for theme
	#rather than for effeciency, in theory I could avoid shuffling
	#and skip straight to dealing 6 random cards to players.
	def Shuffle(self):
		self.hands = [[0]*6,[0]*6]
		self.deckPos = 0

		for i in range(len(self.deck)):
			r = random.randint(0,51)
			temp = self.deck[i]
			self.deck[i] = self.deck[r]
			self.deck[r] = temp
		return self.deck

	def CutToStart(self):
		r1 = random.randint(0, 50)
		r2 = random.randint(r1, 51)
		while self.deck[r1].order == self.deck[r2].order:
			r1 = random.randint(0, 50)
			r2 = random.randint(r1, 51)
		return deck[r1].order > deck[r2].order

	def CutDeck(self, r = -1):
		if r < self.deckPos or r > len(self.deck)-1:
			r = random.randint(self.deckPos,len(self.deck)-1)
		return self.deck[r]

	def DealCards(self):
		for i in range(12):
			self.hands[i%2][i//2] = self.deck[i]
			self.deckPos += 1
		self.hands[0].sort(key = lambda x: x.order)
		self.hands[1].sort(key = lambda x: x.order)

	def ScoreHand(self, hand, cut, crib = False):
		score = 0
		temp_hand = copy.copy(hand)
		#print(len(hand))
		suits = ['heart', 'spade', 'clover', 'diamond']
		#order cut card into hand and check for right jack
		for i in range(len(hand)):
			card = hand[i]
			if len(temp_hand) < 5 and card.order >= cut.order:
				if i == 0:
					temp_hand = [cut] + temp_hand
				else:
					temp_hand = temp_hand[:i] + [cut] + temp_hand[i:]
			if card.order == 11 and card.suit == cut.suit:
				score += 1
		if len(temp_hand) == 4:
			temp_hand = temp_hand + [cut]

		#Find runs, flushes, and pairs
		m = [1]*5 #multiplicity
		run = [1]*5
		flush = [0]*4
		for i in range(len(temp_hand)):
			pair = False
			if i!=0 and temp_hand[i-1].order == temp_hand[i].order:
				m[i] = m[i-1] + 1
				m[i-1] = 1
				pair = True
			if i != 0 and temp_hand[i-1].order +1 == temp_hand[i].order:
				run[i] = run[i-1]+1
			elif pair:
				run[i] = run[i-1]
			flush[suits.index(temp_hand[i].suit)] += 1

		#Count Double Runs and pairs
		pair_scores = [0,2,6,12]
		in_run = False
		run_len = 0
		run_mult = 1
		for i in reversed(range(len(temp_hand))):
			if not in_run and run[i] >= 3:
				in_run = True
			run_len += in_run
			if in_run and m[i] > 1:
				run_len -= m[i] - 1
			run_mult *= max(1, (in_run * m[i]))
			if in_run and run[i] == 1 and m[i] == 1 and m[i+1] == 1:
				in_run = False
			score += pair_scores[m[i]-1] #counts score from pairs

			if i < 4 and flush[i] > 3:
				if not crib and flush[i] >= 4:
					score += flush[i]
				elif crib and flush[i] == 5:
					score += flush[i]
		score += run_len*run_mult

		#Count 15s
		p2 = [1,2,4,8,16]
		for i in range(1,32): #there are 31 subsets of 5 cards if we do not count empty set
			sub = [1 if i & (1 << (7-n)) else 0 for n in range(8)] #converts i into list of binary
			sub = sub[3:]
			temp_sum = 0
			for i in range(len(sub)):
				temp_sum += temp_hand[i].value*sub[i]
			if temp_sum == 15:
				score += 2
		return score

	def FastDeal(self):
		x =  random.sample(self.deck, 5)
		hand, cut = x[:4], x[4]
		hand.sort(key = lambda x: x.order)
		return  hand, cut

	def FastDeal_Dual(self):
		x =  random.sample(self.deck, 13)
		hand1, hand2, cut = x[:6], x[6:12], x[12]
		hand1.sort(key = lambda x: x.order)
		hand2.sort(key = lambda x: x.order)
		return  hand1, hand2, cut

	def Play_Phase(self, card):
		sum = self.playstack[8] + card.value
		index = self.playstack[9]
		start = self.playstack[10]
		score = 0
		#make sure card choice is valid
		if sum > 31:
			return -1
		self.playstack[8] = sum
		#check for 15
		if sum == 15:
			score += 2
		self.playstack[index] = card #add card to stack since we verified sum <= 31
		self.playstack[9] += 1
		#Check for pairs
		score += PlayPairs(self.playstack)
		#Check for runs. Runs do not have to be in order in the stack (e.g. 2,3,1 is a run) so the problem is a little complex
		score += PlayRun(self.playstack)
		
		
		if sum == 31:
			score += 2
			self.Go()
		return score
    
	def Go(self):
		self.playstack[8] = 0
		self.playstack[10] = self.playstack[9]
        
	def ResetPlay(self):
		self.playstack = [0]*11

if __name__ == "__main__":
	test_hand = [Card('heart',1,1), Card('spade', 1, 1), Card('diamond', 1,1), Card('heart', 2, 2)]
	test_cut = Card('heart', 3, 3)

	G = Game()
	x = G.ScoreHand(test_hand, test_cut)
	G.Play_Phase(Card('heart', 1, 1))
	print("Total Score: ",x)
