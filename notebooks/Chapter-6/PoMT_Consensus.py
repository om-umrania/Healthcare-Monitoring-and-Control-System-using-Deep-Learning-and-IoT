import sys
import numpy as np
import pandas as pd
import random
import time
import CryptographicFunctions as cf
import Node

def run_pomt_consensus(numBlocks=1000, maxNodes=50):
    nodeStakes = [1] * maxNodes
    maxInt = sys.maxsize
    blockchain = []
    avgDelay = 0
    ts1 = time.time()
    counter2 = 0

    packets_sent = 0
    packets_received = 0

    for count in range(0, numBlocks):
        src = round(random.random() * (maxNodes-1))
        nodeStake = nodeStakes[src]
        dest = round(random.random() * maxNodes)
        ts = time.time()

        if count == 0:
            prevHash = ''
        else:
            prevHash = blockchain[counter2-1]['hash']

        block = {}
        block['src'] = src
        block['dest'] = dest
        block['ts'] = ts
        block['prevHash'] = prevHash
        block['data'] = "Sample data"  # Add this line
        block['nonce'] = nodeStake + round(random.random() * src) + round(random.random() * ts)

        hash_val = cf.findHash(block)
        isUnique = cf.checkForUniqueHash(blockchain, hash_val)
        counter = 0
        while not isUnique:
            block['nonce'] = nodeStake + round(random.random() * src) + counter + round(random.random() * ts)
            counter += 1
            hash_val = cf.findHash(block)
            isUnique = cf.checkForUniqueHash(blockchain, hash_val)

        nodeStakes[src] = nodeStake + 1

        ts2 = time.time()
        delay = (ts2 - ts)
        avgDelay += delay

        block['hash'] = hash_val

        packets_sent += 1  # Increment packets sent counter
        if isUnique:
            packets_received += 1  # Increment packets received counter

        if count % 2 == 0:
            blockchain.append(block)
            counter2 += 1

        print(f'Processed block {count}, Delay {delay:.06f} s')

    avgDelay /= len(blockchain) if len(blockchain) > 0 else 1

    print(f'PoMT Average delay to mine the block: {avgDelay:.04f} s')
    ts2 = time.time()
    print(f'PoMT Total delay: {ts2-ts1:.04f} s')

    pdr = (packets_received / packets_sent) * 100 if packets_sent > 0 else 0
    print(f'PoMT Packet Delivery Ratio (PDR): {pdr:.2f}%')

    return {
        'avg_delay': avgDelay,
        'total_delay': ts2 - ts1,
        'blockchain': blockchain,
        'pdr': pdr  # Return PDR
    }
