import numpy as np
import pandas as pd
import random
import time
import CryptographicFunctions as cf
import sys
import heartpy as hp
import matplotlib.pyplot as plt
import Node
import random

data_vals, _ = hp.load_exampledata(0)
plt.plot(data_vals)

numBlocks = 1000
maxNodes = 50
nodeStakes = [1] * maxNodes
maxData = len(data_vals)
maxInt = sys.maxsize
blockchain = []
maxLocEnergy = 100

#Simulate an ideal network, attackProb = 0
#Simulate a network with small scale attacks, prob 10% to 20%
attackProb = 0.2 #Percentage of nodes to be attacked

#removeAttack = input('Do you want to remove attack(1-Yes):')
removeAttack = '0'

nodes = []
for count in range(0, maxNodes) :
    loc_x = (random.random()*maxLocEnergy)
    loc_y = (random.random()*maxLocEnergy)
    energy = (random.random()*maxLocEnergy)
    trust = (random.random()*maxLocEnergy)
    
    node = Node.Node(str(count), (loc_x, loc_y), trust, energy)
    #Store blank blockchain in all nodes
    #node.storeBlockchain(blockchain)
    nodes.append(node)

avgDelay = 0
ts1 = time.time()

for count in range(0, numBlocks) :
    src = round(random.random() * (maxNodes-1))
    nodeStake = nodeStakes[src]
    dest = round(random.random() * (maxNodes-1))
    nodes[src].attackNode(attackProb)
    nodes[dest].attackNode(attackProb)
    
    data = round(random.random() * (maxData-1))
    data = data_vals[data]
    ts = time.time()
    
    if(removeAttack == '1') :
        for count2 in range(0, maxNodes) :
            currNode = nodes[count2]
            currNode.correctBlockchain()
    
    #Now check trust levels of nearby nodes
    srcNode = nodes[src]
    avgLevel = 0
    for count2 in range(0, maxNodes) :
        if(count2 == srcNode) :
            continue
        else :
            currNode = nodes[count2]
            avgLevel = avgLevel + srcNode.getTrustScore(currNode)
    
    #Find nodes with good trust levels
    avgLevel = avgLevel / (maxNodes-1)
    nodeIndices = []
    for count2 in range(0, maxNodes) :
        if(count2 == srcNode) :
            continue
        else :
            #Select nodes with higher trust level as miner nodes
            currNode = nodes[count2]
            if(srcNode.getTrustScore(currNode) > avgLevel) :
                nodeIndices.append(count2)
    
    #Select these nodes, and verify their chains for adding the block
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
    
    #Apply Pos+PoW (use stake of src node & random timestamp for mining)
    block['nonce'] = nodeStake + round(random.random() * src) + round(random.random() * ts)
    
    isUnique = True
    for count2 in range(0, len(nodeIndices)) :
        index = nodeIndices[count2]
        isUnique = nodes[index].canAddBlock(block)
        counter = 0
        while(isUnique == False) :
            block['nonce'] = nodeStake + round(random.random() * src) + counter + round(random.random() * ts)
            counter = counter + 1
            isUnique = nodes[index].canAddBlock(block)
    
    nodeStakes[src] = nodeStake + 1
    
    ts2 = time.time()
    delay = (ts2 - ts)
    avgDelay = avgDelay + delay
    
    block['hash'] = cf.findHash(block)
    
    #print('Adding block %d' % (len(blockchain)))
    blockchain.append(block)
    for count2 in range(0, maxNodes) :
        nodes[count2].addBlock(block)
    
    print('Processed block %d, Delay %0.06f s' % (count, delay))
    
avgDelay = avgDelay / len(blockchain)

print('Ensemble Average delay to mine the block: %0.04f s' % (avgDelay))
ts2 = time.time();
print('Ensemble Total delay: %0.04f s' % (ts2-ts1))