import numpy as np
import pandas as pd
import random
import time
import CryptographicFunctions as cf
import sys
import heartpy as hp
import matplotlib.pyplot as plt

def run():
    data_vals, _ = hp.load_exampledata(0)
    # plt.plot(data_vals)
    # plt.show()

    numBlocks = 1000
    maxNodes = 50
    nodeStakes = [1] * maxNodes
    maxData = len(data_vals)
    maxInt = sys.maxsize
    blockchain = []

    avgDelay = 0
    ts1 = time.time()
    counter2 = 0

    for count in range(numBlocks):
        src = round(random.random() * (maxNodes - 1))
        nodeStake = nodeStakes[src]
        dest = round(random.random() * maxNodes)
        data = round(random.random() * (maxData - 1))
        data = data_vals[data]
        ts = time.time()
        
        if count == 0:
            prevHash = ''
        else:
            prevHash = blockchain[counter2 - 1]['hash']
        
        block = {
            'src': src,
            'dest': dest,
            'data': data,
            'ts': ts,
            'prevHash': prevHash,
            'nonce': nodeStake + round(random.random() * src) + round(random.random() * ts)
        }
        
        hash_val = cf.findHash(block)
        isUnique = cf.checkForUniqueHash(blockchain, hash_val)
        counter = 0
        while not isUnique:
            block['nonce'] = nodeStake + round(random.random() * src) + counter + round(random.random() * ts)
            counter += 1
            hash_val = cf.findHash(block)
            isUnique = cf.checkForUniqueHash(blockchain, hash_val)
        
        nodeStakes[src] += 1
        
        ts2 = time.time()
        delay = ts2 - ts
        avgDelay += delay
        
        block['hash'] = hash_val
        
        if count % 2 == 0:
            blockchain.append(block)
            counter2 += 1
        
        print(f'Processed block {count}, Delay {delay:.06f} s')
        
    if len(blockchain) > 0:
        avgDelay = avgDelay / len(blockchain)
        print(f'Ensemble Average delay to mine the block: {avgDelay:.04f} s')
    else:
        print('No blocks were added to the blockchain.')

    ts2 = time.time()
    print(f'Ensemble Total delay: {ts2 - ts1:.04f} s')

    results = {
        "avg_delay": avgDelay,
        "total_delay": ts2-ts1,
        "blockchain": blockchain
    }

if __name__ == "__main__":
    run()