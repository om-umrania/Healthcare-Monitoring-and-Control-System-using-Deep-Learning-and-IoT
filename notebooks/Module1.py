import numpy as np
import pandas as pd
import random
import time
import CryptographicFunctions as cf
import sys
import heartpy as hp
import matplotlib.pyplot as plt

#Read the data samples
data_vals, _ = hp.load_exampledata(0)
plt.plot(data_vals)

#Setup constants
numBlocks = 10000
maxNodes = 50
maxData = len(data_vals)
maxInt = 100
blockchain = []

avgDelay = 0

#Mark the starting timestamp
ts1 = time.time();

for count in range(0, numBlocks) :
    #Setup the block parameters
    src = round(random.random() * maxNodes)
    dest = round(random.random() * maxNodes)
    data = round(random.random() * (maxData-1))
    data = data_vals[data]
    ts = time.time()
    
    #If this is the Genesis block, then mark previous hash as ''
    if(count == 0) :
        prevHash = ''
    else :
        prevHash = blockchain[count-1]['hash']
    
    #Setup the block
    block = {}
    block['src'] = src
    block['dest'] = dest
    block['data'] = data
    block['ts'] = ts
    block['prevHash'] = prevHash
    
    #Apply PoW
    block['nonce'] = round(random.random() * maxInt)
    
	#Find the hash
    hash_val = cf.findHash(block)
    #Check if this hash is unique
    isUnique = cf.checkForUniqueHash(blockchain, hash_val)
    while(isUnique == False) :
		#If not, then generate a new nonce value sets
        block['nonce'] = round(random.random() * maxInt)
        hash_val = cf.findHash(block)
        isUnique = cf.checkForUniqueHash(blockchain, hash_val)
    
    #Find the completion timestamps
    ts2 = time.time()
    delay = (ts2 - ts)
    avgDelay = avgDelay + delay
    
    block['hash'] = hash_val
    
    #Add the block to the chain sets
    blockchain.append(block)
    
    print('Processed block %d, Delay %0.06f s' % (count, delay))
    
avgDelay = avgDelay / len(blockchain)

ts2 = time.time();
print('PoW Total delay: %0.04f s' % (ts2-ts1))
print('PoW Average delay to mine the block: %0.04f s' % (avgDelay))