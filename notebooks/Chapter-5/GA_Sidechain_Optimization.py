import numpy as np
import pandas as pd
import random
import time
import CryptographicFunctions as cf
import heartpy as hp
import matplotlib.pyplot as plt
import Node

def run(show_block_schema=True):
    data_vals, _ = hp.load_exampledata(0)

    Ni = 10  # Number of iterations
    Ns = 10  # Number of solutions
    Lr = 0.95  # Learning rate
    Nsc = 5  # Number of sidechains
    Lsc = 100  # Length of sidechain

    sols_to_change = [1] * Ns
    solution = []
    fitness = []

    sidechains = [[] for _ in range(Nsc)]
    blockchain = []

    maxNodes = 10
    nodeStakes = [random.randint(1, 100) for _ in range(maxNodes)]

    nodes = [Node.Node(str(i), (random.random(), random.random()), random.randint(1, 100), random.random()*100) for i in range(maxNodes)]
    initE = sum(node.energy for node in nodes)

    # Classification Metrics Counters
    TP = FP = TN = FN = 0

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
                        'nonce': nodeStake + random.randint(0, src) + round(random.random() * ts),
                    }
                    block['hash'] = cf.findHash(block)

                    while not dummyNode.canAddBlock(block):
                        block['nonce'] = nodeStake + random.randint(0, src) + round(random.random() * ts)
                        block['hash'] = cf.findHash(block)

                    dummyNode.addBlock(block)

                    # Simulated classification decision
                    actual = "valid" if nodeStake > 25 or np.var(data) > 120 else "invalid"
                    prediction = "valid" if nodeStake > 30 or np.var(data) > 100 else "invalid"
                    
                    if random.random() < 0.1:
                        prediction = "valid" if actual == "invalid" else "invalid"  # flip 10% predictions
                    
                    print("------------------------------------------------------------")
                    print(f"Actual Valid: {(actual == 'valid')}, Predicted Valid: {(prediction == 'valid')}")
                    print("------------------------------------------------------------")


                    if prediction == "valid" and actual == "valid":
                        TP += 1
                    elif prediction == "valid" and actual == "invalid":
                        FP += 1
                    elif prediction == "invalid" and actual == "valid":
                        FN += 1
                    elif prediction == "invalid" and actual == "invalid":
                        TN += 1

                    nodes[src].packets_sent += 1
                    nodes[dest % maxNodes].packets_received += 1

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

    total_packets_sent = sum(node.packets_sent for node in nodes)
    total_packets_received = sum(node.packets_received for node in nodes)
    pdr = (total_packets_received / total_packets_sent) if total_packets_sent > 0 else 0

    print(f'Packet Delivery Ratio (PDR): {pdr:.04f}')

    # Calculate Classification Metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n--- Classification Metrics (Simulated Block Validation) ---")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Negatives (FN): {FN}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1_score:.4f}")
    
    # Visualization of Classification Metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1_score]
    plt.figure()
    plt.bar(metrics, values, color='orange')
    plt.title('Classification Metrics - GA Sidechain Optimization')
    plt.ylabel('Score')
    plt.grid(axis='y')
    plt.ylim(0, 1)
    plt.show()

    return {
        'fitness': fitness,
        'final_energy': finalE,
        'energy_needed': eNeeded,
        'fitness_threshold': fth,
        'pdr': pdr,
        'sidechains': sidechains,
        'classification_metrics': {
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1_score
        }
    }

if __name__ == "__main__":
    run(show_block_schema=False)