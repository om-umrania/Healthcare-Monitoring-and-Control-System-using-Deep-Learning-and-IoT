import GA_Sidechain_Optimization as gso
import IoT_QoS_Optimized_Blockchain as qos_blockchain
import Enhanced_PoS_Consensus as consensus
import matplotlib.pyplot as plt

def main():
    print("Running GA Sidechain Optimization...")
    ga_results = gso.run()

    print("Running QoS Optimized Blockchain...")
    qos_results = qos_blockchain.run()

    print("Running QoS Optimized Blockchain Consensus...")
    consensus_results = consensus.run()

    # Collect and display results
    print("Collecting and displaying results...")

    # Display GA Sidechain Optimization results
    print("\nGA Sidechain Optimization Results:")
    print(f"Final Energy: {ga_results['final_energy']:.4f} mJ")
    print(f"Energy Needed: {ga_results['energy_needed']:.4f} mJ")
    print(f"Fitness Threshold: {ga_results['fitness_threshold']:.4f}")
    
    # Display QoS Optimized Blockchain results
    print("\nQoS Optimized Blockchain Results:")
    print(f"Average Delay: {qos_results['avg_delay']:.4f} s")
    print(f"Total Delay: {qos_results['total_delay']:.4f} s")
    
    # Display QoS Optimized Blockchain Consensus results
    print("\nEnhanced_PoS_Consensus Results:")
    print(f"Average Delay: {consensus_results['avg_delay']:.4f} s")
    print(f"Total Delay: {consensus_results['total_delay']:.4f} s")

    avg_delay = consensus_results['avg_delay']
    total_delay = consensus_results['total_delay']
    pdr = consensus_results['pdr']
    blockchain = consensus_results['blockchain']

    print(f"Packet Delivery Ratio (PDR): {pdr:.4f}")
    print(f"Number of Blocks: {len(blockchain)}")


    # Visualization
    print("Visualizing results...")

    # Plot fitness over iterations for GA
    plt.figure()
    plt.plot(ga_results['fitness'])
    plt.title('Fitness over Iterations - GA Sidechain Optimization')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.show()

    # Histogram of blockchain lengths for QoS Optimized Blockchain
    blockchain_lengths = [len(block) for block in qos_results['blockchain']]
    plt.figure()
    plt.hist(blockchain_lengths, bins=20)
    plt.title('Blockchain Length Distribution - QoS Optimized Blockchain')
    plt.xlabel('Blockchain Length')
    plt.ylabel('Frequency')
    plt.show()

    # Histogram of blockchain lengths for QoS Optimized Blockchain Consensus
    blockchain_lengths_consensus = [len(block) for block in consensus_results['blockchain']]
    plt.figure()
    plt.hist(blockchain_lengths_consensus, bins=20)
    plt.title('Blockchain Length Distribution - QoS Optimized Blockchain Consensus')
    plt.xlabel('Blockchain Length')
    plt.ylabel('Frequency')
    plt.show()

    # Histogram of blockchain lengths for
    plt.figure(figsize=(12, 6))
    delays = [block['ts'] for block in blockchain]
    plt.plot(delays, label='Block Timestamps')
    plt.xlabel('Block Index')
    plt.ylabel('Timestamp')
    plt.title('Block Mining Timestamps')
    plt.legend()

if __name__ == "__main__":
    main()
