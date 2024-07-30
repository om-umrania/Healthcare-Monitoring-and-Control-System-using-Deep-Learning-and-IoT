import numpy as np
import random
import time
import CryptographicFunctions as cf
import Node

def run_gco_optimization(Ni=10, Ns=10, Lr=0.95, Nsc=5, Lsc=100):
    sidechains = [[] for _ in range(Nsc)]
    solution = []
    fitness = []
    sols_to_change = [1] * Ns

    maxNodes = 10
    nodeStakes = [random.randint(1, 100) for _ in range(maxNodes)]
    nodes = [Node.Node(str(i), (random.random(), random.random()), random.randint(1, 100), random.random()*100) for i in range(maxNodes)]
    initE = sum(node.energy for node in nodes)

    for iteration in range(Ni):
        print(f"Iteration {iteration}")
        
        for sol in range(Ns):
            if sols_to_change[sol] == 1:
                print(f"\tSolution {sol}")
                
                scNumber = random.randint(0, Nsc - 1)
                print(f"\t\tSelected sidechain: {scNumber}")
                
                sidechain = sidechains[scNumber]
                dummyNode = Node.Node('0', (0, 0), 0, 0)
                dummyNode.storeBlockchain(sidechain)
                
                t1 = time.time()
                for blockCount in range(len(sidechain), len(sidechain) + Lsc):
                    prevHash = sidechain[blockCount - 1]['hash'] if blockCount > 0 else ''
                    src = random.randint(0, maxNodes - 1)
                    nodeStake = nodeStakes[src]
                    dest = random.randint(0, maxNodes - 1)
                    ts = time.time()
                    
                    block = {
                        'src': src,
                        'dest': dest,
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

    finalE = sum(node.energy for node in nodes)
    eNeeded = initE - finalE
    print(f"Energy needed: {eNeeded:.04f} mJ")

    return {
        'fitness': fitness,
        'sidechains': sidechains,
        'energy_needed': eNeeded
    }
