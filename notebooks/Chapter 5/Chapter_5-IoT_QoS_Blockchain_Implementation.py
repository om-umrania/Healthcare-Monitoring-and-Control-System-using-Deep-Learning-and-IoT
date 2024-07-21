import CryptographicFunctions as cf
from Node import Node
import GA_Sidechain_Optimization as gso
import IoT_QoS_Optimized_Blockchain as pos
import Enhanced_PoS_Consensus as epc

def main():
    # Initialize Cryptographic Functions
    print("Initializing Cryptographic Functions...")
    # This is to ensure all functions are imported and ready

    # Initialize Nodes
    print("Defining Node Class...")

    # Run GA Sidechain Optimization
    print("Running GA Sidechain Optimization...")
    gso.run()

    # Run IoT QoS Optimized Blockchain (PoS Initialization)
    print("Running IoT QoS Optimized Blockchain...")
    pos.run()

    # Run Enhanced PoS Consensus
    print("Running Enhanced PoS Consensus...")
    epc.run()

    # Collect and display results
    print("Collecting and displaying results...")
    # Add your code for data collection and visualization here

if __name__ == "__main__":
    main()


