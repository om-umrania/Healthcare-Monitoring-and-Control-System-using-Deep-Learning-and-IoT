import numpy as np
import pandas as pd
import random
import time
import CryptographicFunctions as cf
import sys
import heartpy as hp
import matplotlib.pyplot as plt
import Node

def run():
    # Load example data
    data_vals, _ = hp.load_exampledata(0)
    # plt.plot(data_vals)
    # plt.show()

    # GA Parameters
    Ni = 10  # Number of iterations
    Ns = 10  # Number of solutions
    Lr = 0.95  # Learning rate
    Nsc = 5  # Number of sidechains
    Lsc = 100  # Length of sidechain

    # Initialize solutions
    sols_to_change = [1] * Ns
    solution = []  # Initialize the solution list
    fitness = []

    # Sidechain and blockchain initialization
    sidechains = [[] for _ in range(Nsc)]
    blockchain = []

    # Initialize nodes and their stakes
    maxNodes = 10
    nodeStakes = [random.randint(1, 100) for _ in range(maxNodes)]

    # Initialize nodes for energy calculation (assuming a Node class with energy attribute)
    nodes = [Node.Node(str(i), (random.random(), random.random()), random.randint(1, 100), random.random()*100) for i in range(maxNodes)]
    initE = sum(node.energy for node in nodes)  # Initial energy

    # Main Loop for Block Addition
    numBlocks = 1000
    avgDelay = 0
    ts1 = time.time()

    for count in range(numBlocks):
        src = round(random.random() * (maxNodes - 1))
        dest = round(random.random() * maxNodes)
        data = data_vals[round(random.random() * (len(data_vals) - 1))]
        nodeStake = nodeStakes[src]
        ts = time.time()

        if count == 0:
            prevHash = ''
        else:
            prevHash = blockchain[count - 1]['hash']

        block = {
            'src': src,
            'dest': dest,
            'data': data,
            'ts': ts,
            'prevHash': prevHash,
            'nonce': nodeStake + round(random.random() * src)
        }

        hash_val = cf.findHash(block)
        isUnique = cf.checkForUniqueHash(blockchain, hash_val)
        counter = 0
        while not isUnique:
            block['nonce'] = nodeStake + round(random.random() * src) + counter
            counter += 1
            hash_val = cf.findHash(block)
            isUnique = cf.checkForUniqueHash(blockchain, hash_val)
        nodeStakes[src] = nodeStake + 1

        ts2 = time.time()
        delay = ts2 - ts
        avgDelay += delay

        block['hash'] = hash_val
        blockchain.append(block)
        print(f'Processed block {count}, Delay {delay:.06f} s')

    avgDelay /= len(blockchain)
    print(f'PoS Average delay to mine the block: {avgDelay:.04f} s')
    ts2 = time.time()
    print(f'PoS Total delay: {ts2 - ts1:.04f} s')

    # GA & Sidechaining for QoS Optimization
    for iteration in range(Ni):
        print(f"Iteration {iteration}")
        
        for sol in range(Ns):
            if sols_to_change[sol] == 1:
                print(f"\tSolution {sol}")
                
                # Select sidechain at random
                scNumber = random.randint(0, Nsc - 1)
                print(f"\t\tSelected sidechain: {scNumber}")
                
                # Add blocks and evaluate fitness
                sidechain = sidechains[scNumber]
                dummyNode = Node.Node('0', (0, 0), 0, 0)
                dummyNode.storeBlockchain(sidechain)
                
                t1 = time.time()
                for blockCount in range(len(sidechain), len(sidechain) + Lsc):
                    prevHash = sidechain[blockCount - 1]['hash'] if blockCount > 0 else ''
                    src = random.randint(0, maxNodes - 1)
                    nodeStake = nodeStakes[src]
                    dest = random.randint(0, maxNodes - 1)
                    data = data_vals[random.randint(0, len(data_vals) - 1)]
                    ts = time.time()
                    
                    block = {
                        'src': src,
                        'dest': dest,
                        'data': data,
                        'ts': ts,
                        'prevHash': prevHash,
                        'nonce': nodeStake + random.randint(0, src) + round(random.random() * ts),
                    }
                    block['hash'] = cf.findHash(block)
                    
                    while not dummyNode.canAddBlock(block):
                        block['nonce'] = nodeStake + random.randint(0, src) + round(random.random() * ts)
                        block['hash'] = cf.findHash(block)
                    
                    dummyNode.addBlock(block)
                
                t2 = time.time()
                delay = t2 - t1
                
                print(f"\t\tBlock Time: {delay:.04f} seconds")
                print(f"\t\tTPS: {1 / delay:.04f} transactions per second")
                print(f"\t\tThroughput: {len(sidechain) / delay:.04f} kbps")
                
                if iteration == 0:
                    solution.append(scNumber)
                    fitness.append(delay)
                else:
                    solution[sol] = scNumber
                    fitness[sol] = delay
        
        fth = sum(fitness) * Lr / len(fitness)
        print(f"Fitness threshold: {fth:.04f}")
        
        for sol in range(Ns):
            sols_to_change[sol] = 1 if fitness[sol] >= fth else 0

    # Final energy calculation and summary
    finalE = sum(node.energy for node in nodes)
    eNeeded = initE - finalE
    print(f"Energy needed: {eNeeded:.04f} mJ")

    results = {
        "avg_delay": avgDelay,
        "total_delay": ts2-ts1,
        "blockchain": blockchain
    }
    return results

if __name__ == "__main__":
    run()