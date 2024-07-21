# import CryptographicFunctions as cf
# from Node import Node
# import GA_Sidechain_Optimization as gso
# import IoT_QoS_Optimized_Blockchain as pos
# import Enhanced_PoS_Consensus as epc

# def main():
#     # Initialize Cryptographic Functions
#     print("Initializing Cryptographic Functions...")
#     # This is to ensure all functions are imported and ready

#     # Initialize Nodes
#     print("Defining Node Class...")

#     # Run GA Sidechain Optimization
#     print("Running GA Sidechain Optimization...")
#     gso.run()

#     # Run IoT QoS Optimized Blockchain (PoS Initialization)
#     print("Running IoT QoS Optimized Blockchain...")
#     pos.run()

#     # Run Enhanced PoS Consensus
#     print("Running Enhanced PoS Consensus...")
#     epc.run()

#     # Collect and display results
#     print("Collecting and displaying results...")
#     # Add your code for data collection and visualization here

# if __name__ == "__main__":
#     main()


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
    print("\nQoS Optimized Blockchain Consensus Results:")
    print(f"Average Delay: {consensus_results['avg_delay']:.4f} s")
    print(f"Total Delay: {consensus_results['total_delay']:.4f} s")

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

if __name__ == "__main__":
    main()
