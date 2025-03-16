import numpy as np
import pandas as pd
import random
import time
import CryptographicFunctions as cf
import sys
import heartpy as hp
import matplotlib.pyplot as plt
import Node

def run(show_block_schema=True):
    print("[DEBUG] run() function inside IoT_QoS_Optimized_Blockchain is now executing...")

    data_vals, _ = hp.load_exampledata(0)

    Ni = 10
    Ns = 10
    Lr = 0.95
    Nsc = 5
    Lsc = 100

    sols_to_change = [1] * Ns
    solution = []
    fitness = []

    sidechains = [[] for _ in range(Nsc)]
    blockchain = []

    maxNodes = 10
    nodeStakes = [random.randint(1, 100) for _ in range(maxNodes)]
    nodes = [Node.Node(str(i), (random.random(), random.random()), random.randint(1, 100), random.random()*100) for i in range(maxNodes)]
    initE = sum(node.energy for node in nodes)

    numBlocks = 1000
    avgDelay = 0
    ts1 = time.time()

    # Classification Counters
    TP = FP = TN = FN = 0
    delays_list = []

    for count in range(numBlocks):
        src = round(random.random() * (maxNodes - 1))
        dest = round(random.random() * maxNodes)
        data = data_vals[round(random.random() * (len(data_vals) - 1))]
        nodeStake = nodeStakes[src]
        ts = time.time()
        prevHash = blockchain[count - 1]['hash'] if count > 0 else ''

        sensor_type = "ECG"
        patient_info = {
            "name": "John Doe",
            "address": "Ward 3B, Room 202",
            "contact": "+91-XXXXX-XXXXX"
        }
        doctor_info = {
            "name": "Dr. A Sharma",
            "specialty": "Cardiologist",
            "contact": "doctor@hospital.com"
        }
        sidechain_id = count % Nsc

        block = {
            'src': src,
            'dest': dest,
            'data': data,
            'sensor_type': sensor_type,
            'patient_info': patient_info,
            'doctor_info': doctor_info,
            'ts': ts,
            'prevHash': prevHash,
            'sidechain_id': sidechain_id,
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

        nodeStakes[src] += 1
        block['hash'] = hash_val

        ts2 = time.time()
        delay = ts2 - ts
        delays_list.append(delay)
        avgDelay += delay

        blockchain.append(block)
        nodes[src].packets_sent += 1
        nodes[dest % maxNodes].packets_received += 1

        if show_block_schema:
            print("Block Schema and Data:")
            for key, value in block.items():
                print(f"  {key}: {value}")
            print("-----------------------------")

        print(f'Processed block {count}, Delay {delay:.06f} s')

    qos_threshold = np.percentile(delays_list, 90)

    for count in range(len(blockchain)):
        nodeStake = blockchain[count]['nonce']
        delay = delays_list[count]

        predicted_class = "good" if nodeStake > 50 else "poor"
        actual_class = "good" if delay <= qos_threshold else "poor"

        if predicted_class == "good" and actual_class == "good":
            TP += 1
        elif predicted_class == "good" and actual_class == "poor":
            FP += 1
        elif predicted_class == "poor" and actual_class == "good":
            FN += 1
        elif predicted_class == "poor" and actual_class == "poor":
            TN += 1

    avgDelay /= len(blockchain)
    print(f'PoS Average delay to mine the block: {avgDelay:.04f} s')
    ts2 = time.time()
    print(f'PoS Total delay: {ts2 - ts1:.04f} s')

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
                    data = data_vals[random.randint(0, len(data_vals) - 1)]
                    ts = time.time()

                    block = {
                        'src': src,
                        'dest': dest,
                        'data': data,
                        'sensor_type': sensor_type,
                        'patient_info': patient_info,
                        'doctor_info': doctor_info,
                        'ts': ts,
                        'prevHash': prevHash,
                        'sidechain_id': scNumber,
                        'nonce': nodeStake + random.randint(0, src) + round(random.random() * ts)
                    }
                    block['hash'] = cf.findHash(block)
                    while not dummyNode.canAddBlock(block):
                        block['nonce'] = nodeStake + random.randint(0, src) + round(random.random() * ts)
                        block['hash'] = cf.findHash(block)

                    dummyNode.addBlock(block)
                    if show_block_schema:
                        print("Block Schema and Data:")
                        for key, value in block.items():
                            print(f"  {key}: {value}")
                        print("-----------------------------")

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

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP+TN+FP+FN) else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n--- QoS Classification Evaluation ---")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1_score:.4f}")

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1_score]
    plt.figure()
    plt.bar(metrics, values, color='orange')
    plt.title('Classification Metrics - IoT QoS Optimized Blockchain')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.show()

    results = {
        "avg_delay": avgDelay,
        "total_delay": ts2 - ts1,
        "blockchain": blockchain,
        "sidechains": sidechains,
        "classification_metrics": {
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1_score
        }
    }
    return results

if __name__ == "__main__":
    run(show_block_schema=True)