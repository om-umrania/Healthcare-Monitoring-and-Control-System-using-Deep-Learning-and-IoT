import warnings
import classification_functions as cf
import DeepLearningProcess as dlp

warnings.filterwarnings("ignore")

dataset_files = [0] * 3
start_rows = [0] * 3
end_rows = [0] * 3
classes_arr = [0] * 3

#Cancer set
dataset_files[0] = 'Cancer/cancer.csv'
start_rows[0] = 1
end_rows[0] = 23
classes_arr[0] = 24

#Parkinsons
dataset_files[1] = 'Parkinson/Parkinson.csv'
start_rows[1] = 0
end_rows[1] = 7
classes_arr[1] = 8

#Blood Cancer
dataset_files[2] = 'Blood/Blood.csv'
start_rows[2] = 0
end_rows[2] = 14
classes_arr[2] = 15

#Find the values of Di
di_arr = [0]*3
for count in range(0,len(dataset_files)) :
    di_arr2 = [0] * (end_rows[count] - start_rows[count])
    out_count = 0
    for count2 in range(start_rows[count], end_rows[count]) :
        di = cf.findAccuracy(dataset_files[count], count2, count2+1, classes_arr[count])
        di_arr2[out_count] = di
        out_count = out_count + 1
    
    di_arr[count] = di_arr2
    
#Find the Dij array
dij_arr = []
for i in range(0,len(di_arr)) :
    for k in range(0,len(di_arr[i])) :
        for j in range(0,len(di_arr)) :
            dij_vals = [0] * len(di_arr[j])
            for l in range(0,len(di_arr[j])) :
                dij = (di_arr[i][k] + di_arr[j][l])/2
                dij = (dij + di_arr[i][k] + di_arr[j][l]) / 3
                dij_vals[l] = dij
                print('D(%d, %d, %d, %d):%0.04f' % (i,k,j,l,dij))
            dij_arr.extend(dij_vals)
#Now use these as weights, and classify the data
print('***********************************')
print('Processing CANCER Dataset')
dlp.applyDL(dataset_files[0])
acc1 = cf.findAccuracy(dataset_files[0], start_rows[0], end_rows[0], classes_arr[0],1,dij_arr)
print('Processing Parkinsons Dataset')
dlp.applyDL(dataset_files[1])
acc2 = cf.findAccuracy(dataset_files[1], start_rows[1], end_rows[1], classes_arr[1],1,dij_arr)
print('Processing Blood Cancer Dataset')
dlp.applyDL(dataset_files[2])
acc3 = cf.findAccuracy(dataset_files[2], start_rows[2], end_rows[2], classes_arr[2],1,dij_arr)
acc = max([acc1,acc2,acc3]);
print('Final accuracy of ensemble learning %0.04f %%' % (acc * 100))

input('Press Y for secure storage...')

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
#import ECCLibrary as ecc

data_vals = dij_arr
plt.plot(data_vals)

#These many blocks will be added to the system
numBlocks = 1000
totBlocks = 0

#Add these many blocks for sidechain checks
numBlocksForSidechainChecks = 500

maxNodes = 50
nodeStakes = [1] * maxNodes
maxData = len(data_vals)
maxInt = sys.maxsize
blockchain = []
eccChain = []

maxLocEnergy = 100

#Simulate an ideal network, attackProb = 0
#Simulate a network with small scale attacks, prob 10% to 20%
attackProb = 0.2 #Percentage of nodes to be attacked

#removeAttack = input('Do you want to remove attack(1-Yes):')
removeAttack = '0'

nodes = []
initE = 0

for count in range(0, maxNodes) :
    loc_x = (random.random()*maxLocEnergy)
    loc_y = (random.random()*maxLocEnergy)
    energy = (random.random()*maxLocEnergy)
    
    initE = initE + energy
    
    trust = (random.random()*maxLocEnergy)
    
    node = Node.Node(str(count), (loc_x, loc_y), trust, energy)
    #Store blank blockchain in all nodes
    #node.storeBlockchain(blockchain)
    nodes.append(node)

avgDelay = 0
sidechains = []

#This loop will assist in generating number of blocks
while(totBlocks < numBlocks) :
    
    ts1 = time.time()
    for count in range(0, numBlocksForSidechainChecks) :
        totBlocks = totBlocks + 1
        
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
                #currNode.correctBlockchain()
        
        #Now check trust levels of nearby nodes
        srcNode = nodes[src]
        avgLevel = 0
        for count2 in range(0, maxNodes) :
            if(count2 == srcNode) :
                continue
            else :
                currNode = nodes[count2]
                avgLevel = avgLevel + srcNode.getTrustScore(currNode, False)
        
        #Find nodes with good trust levels
        avgLevel = avgLevel / (maxNodes-1)
        nodeIndices = []
        for count2 in range(0, maxNodes) :
            if(count2 == srcNode) :
                continue
            else :
                #Select nodes with higher trust level as miner nodes
                currNode = nodes[count2]
                if(srcNode.getTrustScore(currNode, False) > avgLevel) :
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
        #eccChain.append(ecc.eccEncrypt(block))
        
        for count2 in range(0, maxNodes) :
            nodes[count2].addBlock(block)
        
        print('Processed block %d, Delay %0.06f s' % (count, delay))
        
    avgDelay = avgDelay / len(blockchain)
    
    print('Ensemble Average delay to mine the block: %0.04f s' % (avgDelay))
    ts2 = time.time();
    print('Ensemble Total delay: %0.04f s' % (ts2-ts1))
    
    #GA Based Sidechaining
    Ni = 10
    Ns = 10
    Lr = 0.95
    
    solution = []
    fitness = []
    sols_to_change = [1] * Ns
    
    blockchainLength = len(blockchain)
    if(len(sidechains) == 0) :
        #Initially break the chain into 2 parts
        #CREATION OF SIDECHAINS
        sidechains.append(nodes[count2].blockchain[0:round(blockchainLength/2)])
        sidechains.append(nodes[count2].blockchain[round(blockchainLength/2):])
        
        blockchain = sidechains[0]
    
    #Maximum number of dummy blocks to be added
    #Number of elephants EHO
    elephantsToProcess = 100
    
    for iteration in range(0, Ni) :
        print('Iteration %d' % (iteration))
        for sol in range(0, Ns) :
            if(sols_to_change[sol] == 1) :
                print('\t Solution %d' % (sol))
                #Stochastically find a sidehchain
                scNumber = round(random.random() * len(sidechains))
                while(scNumber >= len(sidechains)) :
                    scNumber = round(random.random() * len(sidechains))
                
                print('\t\tSelected sidechain:%d' % (scNumber))
                #Add blocks to this chain, and evaluate its performance
                sidechain = sidechains[scNumber]
                
                dummyNode = Node.Node('0', (0, 0), 0, 0)
                dummyNode.storeBlockchain(sidechain)
                
                t1 = time.time()
                for blockCount in range(len(sidechain), len(sidechain)+elephantsToProcess) :
                    prevHash = sidechain[blockCount-1]['hash']
                    src = round(random.random() * (maxNodes-1))
                    nodeStake = nodeStakes[src]
                    dest = round(random.random() * (maxNodes-1))
                    data = round(random.random() * (maxData-1))
                    data = data_vals[data]
                    ts = time.time()
                    
                    block = {}
                    block['src'] = src
                    block['dest'] = dest
                    block['data'] = data
                    block['ts'] = ts
                    block['prevHash'] = prevHash
                    block['nonce'] = nodeStake + round(random.random() * src) + round(random.random() * ts)
                    block['hash'] = cf.findHash(block)
                    
                    while(dummyNode.canAddBlock(block) == False) :
                        block['nonce'] = nodeStake + round(random.random() * src) + round(random.random() * ts)
                        block['hash'] = cf.findHash(block)
                    
                    dummyNode.addBlock(block)
                    
                t2 = time.time()
                delay = t2 - t1
                
                print('\t\tBlock Time:%0.04f' % (delay))
                print('\t\tTPS:%0.04f kbps' % (1/delay))
                print('\t\tThroughput:%0.04f kbps' % (len(block)/delay))
                
                #Add this to the solution
                if(iteration == 0) :
                    solution.append(scNumber)
                    fitness.append(delay)
                else :
                    solution[sol] = scNumber
                    fitness[sol] = delay
            
        #Check fitness threshold
        fth = sum(fitness) * Lr / len(fitness)
        print('Fitness threshold:%0.04f' % (fth))
        
        for sol in range(0, Ns) :
            #Change the solutions that require large delays
            if(fitness[sol] >= fth) :
                sols_to_change[sol] = 1
            else :
                sols_to_change[sol] = 0
                
    #Identify solution with highest fitness
    maxIndex = 0
    minFitness = 0
    
    for sol in range(0, Ns) :
        if(sol == 0) :
            minFitness = fitness[sol]
            minIndex = solution[sol]
        elif(fitness[sol] < minFitness) :
            minFitness = fitness[sol]
            minIndex = solution[sol]
            
    selectedChain = solution[minIndex]
    print('Selected chain %d' % (selectedChain))
    
    fth = sum(fitness) * Lr * 0.9 / len(fitness)
    
    if(minFitness < fth) :
        print('Splitting current sidechain into 2 intra sidechains...')
        
        #Split the chain into 2 parts
        selChain = sidechains[minIndex]
        
        chain1 = selChain[0:round(len(selChain)/2)]
        chain2 = selChain[round(len(selChain)/2):]
        
        if(len(chain1) > len(chain2)) :
            selChain = chain2
            sidechains[minIndex] = chain1
        else :
            selChain = chain1
            sidechains[minIndex] = chain2
        
        sidechains.append(selChain)
        
        #Now update this chain on each node
        print('Updating this sidechain to all nodes')
        for count2 in range(0, maxNodes) :
            nodes[count2].storeBlockchain(selChain)
        
        blockchain = selChain

finalE = 0
for count in range(0, maxNodes) :
    finalE = finalE + nodes[count].energy
    

eNeeded = initE - finalE
print('Energy needed %0.04f mJ' % (eNeeded))

print('\t\tBlock Time:%0.04f' % (delay))
print('\t\tTPS:%0.04f kbps' % (1/delay))
print('\t\tThroughput:%0.04f kbps' % (len(block)/delay))