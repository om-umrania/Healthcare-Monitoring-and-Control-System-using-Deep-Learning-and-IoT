import numpy as np
import pandas as pd
import random
import time
import CryptographicFunctions as cf
import heartpy as hp
import matplotlib.pyplot as plt
import Node
import sys

def run():
    data_vals, _ = hp.load_exampledata(0)
    
    numBlocks = 1000
    maxNodes = 50
    nodeStakes = [1] * maxNodes
    maxData = len(data_vals)
    maxInt = sys.maxsize
    blockchain = []

    avgDelay = 0
    ts1 = time.time()
    counter2 = 0

    nodes = [Node.Node(str(i), (random.random(), random.random()), random.randint(1, 100), random.random() * 100) for i in range(maxNodes)]

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

        block = {}
        block['src'] = src
        block['dest'] = dest
        block['data'] = data
        block['ts'] = ts
        block['prevHash'] = prevHash

        # Apply Pos+PoW (use stake of src node & random timestamp for mining)
        block['nonce'] = nodeStake + round(random.random() * src) + round(random.random() * ts)

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
        delay = (ts2 - ts)
        avgDelay += delay

        block['hash'] = hash_val

        if count % 2 == 0:
            blockchain.append(block)
            counter2 += 1

        # Track packets sent and received
        nodes[src].packets_sent += 1
        nodes[dest % maxNodes].packets_received += 1

        print(f'Processed block {count}, Delay {delay:.06f} s')

    avgDelay = avgDelay / len(blockchain) if len(blockchain) > 0 else 0

    print(f'Ensemble Average delay to mine the block: {avgDelay:.04f} s')
    ts2 = time.time()
    print(f'Ensemble Total delay: {ts2 - ts1:.04f} s')

    # Calculate PDR
    total_packets_sent = sum(node.packets_sent for node in nodes)
    total_packets_received = sum(node.packets_received for node in nodes)
    pdr = (total_packets_received / total_packets_sent) if total_packets_sent > 0 else 0

    print(f'Packet Delivery Ratio (PDR): {pdr:.04f}')

    # Return results for visualization
    return {
        'avg_delay': avgDelay,
        'total_delay': ts2 - ts1,
        'pdr': pdr,
        'blockchain': blockchain
    }
