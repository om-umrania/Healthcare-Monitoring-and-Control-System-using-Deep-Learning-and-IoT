import numpy as np
import pandas as pd
import random
import time
import CryptographicFunctions as cf
import sys
import heartpy as hp
import matplotlib.pyplot as plt

data_vals, _ = hp.load_exampledata(0)
plt.plot(data_vals)

numBlocks = 1000
maxNodes = 50
nodeStakes = [1] * maxNodes
maxData = len(data_vals)
maxInt = sys.maxsize
blockchain = []

avgDelay = 0
ts1 = time.time();

for count in range(0, numBlocks) :
    src = round(random.random() * (maxNodes-1))
    dest = round(random.random() * maxNodes)
    data = round(random.random() * (maxData-1))
    data = data_vals[data]
    nodeStake = nodeStakes[src]
    ts = time.time()
    
    if(count == 0) :
        prevHash = ''
    else :
        prevHash = blockchain[count-1]['hash']
    
    block = {}
    block['src'] = src
    block['dest'] = dest
    block['data'] = data
    block['ts'] = ts
    block['prevHash'] = prevHash
    
    #Apply PoS (use stake of src node for mining)
    block['nonce'] = nodeStake + round(random.random() * src)
    
    hash_val = cf.findHash(block)
    isUnique = cf.checkForUniqueHash(blockchain, hash_val)
    counter = 0
    while(isUnique == False) :
        block['nonce'] = nodeStake + round(random.random() * src) + counter
        counter = counter + 1
        hash_val = cf.findHash(block)
        isUnique = cf.checkForUniqueHash(blockchain, hash_val)
    nodeStakes[src] = nodeStake + 1
    
    ts2 = time.time()
    delay = (ts2 - ts)
    avgDelay = avgDelay + delay
    
    block['hash'] = hash_val
    
    blockchain.append(block)
    
    print('Processed block %d, Delay %0.06f s' % (count, delay))
    
avgDelay = avgDelay / len(blockchain)

print('PoS Average delay to mine the block: %0.04f s' % (avgDelay))
ts2 = time.time();
print('PoS Total delay: %0.04f s' % (ts2-ts1))