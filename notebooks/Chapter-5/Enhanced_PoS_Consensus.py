import numpy as np
import pandas as pd
import random
import time
import CryptographicFunctions as cf
import heartpy as hp
import matplotlib.pyplot as plt
import Node
import sys

def generate_patient_info():
    names = ["John Doe", "Jane Smith", "Arjun Patel", "Li Wei", "Carlos Martinez"]
    addresses = ["Ward 1A, Room 101", "Ward 2B, Room 202", "Ward 3C, Room 303", "Ward 4D, Room 404", "Ward 5E, Room 505"]
    contacts = ["+91-9876543210", "+91-9123456780", "+91-9001234567", "+91-9988776655", "+91-8899776655"]
    return {
        "name": random.choice(names),
        "address": random.choice(addresses),
        "contact": random.choice(contacts)
    }

def generate_doctor_info():
    names = ["Dr. A Sharma", "Dr. R Mehta", "Dr. L Chen", "Dr. S Kapoor", "Dr. J Singh"]
    specialties = ["Cardiologist", "Neurologist", "Pulmonologist", "Oncologist", "Endocrinologist"]
    contacts = ["doc1@hospital.com", "doc2@hospital.com", "doc3@hospital.com", "doc4@hospital.com", "doc5@hospital.com"]
    return {
        "name": random.choice(names),
        "specialty": random.choice(specialties),
        "contact": random.choice(contacts)
    }

def run(show_block_schema=True):
    data_vals, _ = hp.load_exampledata(0)

    numBlocks = 1000
    maxNodes = 50
    nodeStakes = [random.randint(10, 100) for _ in range(maxNodes)]
    maxData = len(data_vals)
    blockchain = []
    block_delays = []
    trust_scores = []

    avgDelay = 0
    ts1 = time.time()
    counter2 = 0

    nodes = [Node.Node(str(i), (random.random(), random.random()), nodeStakes[i], random.uniform(50, 150)) for i in range(maxNodes)]

    TP = FP = TN = FN = 0

    for count in range(numBlocks):
        src = random.randint(0, maxNodes - 1)
        nodeStake = nodeStakes[src]
        dest = random.randint(0, maxNodes - 1)
        data = data_vals[random.randint(0, maxData - 1)]
        ts = time.time()

        prevHash = '' if count == 0 else blockchain[counter2 - 1]['hash']
        sensor_type = "ECG"
        patient_info = generate_patient_info()
        doctor_info = generate_doctor_info()

        sidechain_id = count % 5
        nonce = nodeStake + round(random.random() * src) + round(random.random() * ts)

        block = {
            "src": src,
            "dest": dest,
            "data": data,
            "sensor_type": sensor_type,
            "patient_info": patient_info,
            "doctor_info": doctor_info,
            "ts": ts,
            "prevHash": prevHash,
            "sidechain_id": sidechain_id,
            "nonce": nonce
        }

        hash_val = cf.findHash(block)
        isUnique = cf.checkForUniqueHash(blockchain, hash_val)
        counter = 0
        while not isUnique:
            block['nonce'] = nodeStake + round(random.random() * src) + counter + round(random.random() * ts)
            counter += 1
            hash_val = cf.findHash(block)
            isUnique = cf.checkForUniqueHash(blockchain, hash_val)

        block['hash'] = hash_val
        nodeStakes[src] += 1

        ts2 = time.time()
        delay = ts2 - ts
        avgDelay += delay
        block_delays.append(delay)

        # Enhanced Trust Score Logic
        energy_weight = nodes[src].energy / max(node.energy for node in nodes)
        stake_weight = nodeStake / max(nodeStakes)
        pdr_weight = nodes[src].packets_received / (nodes[src].packets_sent + 1)
        trust_score = 0.2 * energy_weight + 0.5 * stake_weight + 0.3 * pdr_weight
        trust_scores.append(trust_score)

        if count % 2 == 0:
            blockchain.append(block)
            counter2 += 1

        nodes[src].packets_sent += 1
        nodes[dest % maxNodes].packets_received += 1

        print(f'Processed block {count}, Delay {delay:.06f} s')
        if show_block_schema:
            print("Block Schema and Data:")
            for key, value in block.items():
                print(f"  {key}: {value}")
            print("-----------------------------")

    avgDelay = avgDelay / len(blockchain) if len(blockchain) > 0 else 0
    ts2 = time.time()

    print(f'Ensemble Average delay to mine the block: {avgDelay:.04f} s')
    print(f'Ensemble Total delay: {ts2 - ts1:.04f} s')

    total_packets_sent = sum(node.packets_sent for node in nodes)
    total_packets_received = sum(node.packets_received for node in nodes)
    pdr = (total_packets_received / total_packets_sent) if total_packets_sent > 0 else 0
    print(f'Packet Delivery Ratio (PDR): {pdr:.04f}')

    # Dynamic thresholds
    trust_threshold = np.percentile(trust_scores, 70)  # Top 30% trusted
    qos_threshold = np.percentile(block_delays, 70)   # Top 30% worst delay = faulty

    for i in range(len(trust_scores)):
        predicted_class = "secure" if trust_scores[i] >= trust_threshold else "faulty"
        actual_class = "secure" if block_delays[i] <= qos_threshold else "faulty"

        if predicted_class == "secure" and actual_class == "secure":
            TP += 1
        elif predicted_class == "secure" and actual_class == "faulty":
            FP += 1
        elif predicted_class == "faulty" and actual_class == "secure":
            FN += 1
        elif predicted_class == "faulty" and actual_class == "faulty":
            TN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n--- Block Trust Classification Evaluation (Enhanced) ---")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1_score:.4f}")

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1_score]
    plt.figure()
    plt.bar(metrics, values, color='orange')
    plt.title('Classification Metrics - Enhanced PoS Consensus (Trust Score Logic)')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.show()

    return {
        'avg_delay': avgDelay,
        'total_delay': ts2 - ts1,
        'pdr': pdr,
        'classification_metrics': {
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1_score
        },
        'blockchain': blockchain
    }

if __name__ == "__main__":
    run(show_block_schema=True)