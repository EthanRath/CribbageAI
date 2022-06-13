import numpy as np
import copy
from Cribbage import Card


def PlayRun(playstack):
    run = 0
    index = playstack[9]
    start = playstack[10]
    for i in range(1,index - start+1):
        temp = copy.copy(playstack[index - i:index])
        temp.sort(key = lambda x: x.order)
        success = True
        for j in range(len(temp)):
            if j != 0 and temp[j-1].order != temp[j].order -1:
                success = False
        if success:
            run = i
    return run * (run >= 3)

def PlayPairs(playstack):
    score = 0
    index = playstack[9]
    mult = 0
    for i in reversed(range(1,index)):
        if playstack[i].order == playstack[i-1].order:
            mult += 1
        else:
            break
    #score pairs
    if mult >= 1:
        pair_scores = [2,6,12]
        score = pair_scores[mult-1]
    return score

if __name__ == "__main__":
    teststack = [Card('heart',4,4), Card('spade', 2, 2), Card('diamond', 2,2), 0, 0, 0, 0, 0, 3, 3, 0]
    print(PlayPairs(teststack))
    print("Break")
    teststack = [Card('heart',5,5), Card('spade', 4, 4), Card('diamond', 3,3), Card("diamond", 2, 2), 0, 0, 0, 0, 3, 4, 1]
    print(PlayRun(teststack))
