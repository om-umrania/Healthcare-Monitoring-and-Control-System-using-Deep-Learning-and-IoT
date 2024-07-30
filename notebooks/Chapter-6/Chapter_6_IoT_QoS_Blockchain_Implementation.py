import PoMT_Consensus as pomt
import Genetic_Chain_Optimization as gco
import DeepLearningModels as dlm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

def main():
    print("Running PoMT Consensus...")
    pomt_results = pomt.run_pomt_consensus()
    
    print("Running GA Sidechain Optimization...")
    gco_results = gco.run_gco_optimization()
    
    # Define rnn_data and rnn_labels
    rnn_data = np.random.rand(100, 10, 1)  # Example data, replace with actual data
    rnn_labels = np.random.randint(2, size=(100, 1))  # Example labels, replace with actual labels
    
    print("Training RNN Model...")
    rnn_model, rnn_history = dlm.train_rnn_model(rnn_data, rnn_labels)
    
    # Define cnn_data and cnn_labels
    cnn_data = np.random.rand(100, 64, 64, 1)  # Example data, replace with actual data
    cnn_labels = np.random.randint(2, size=(100, 1))  # Example labels, replace with actual labels
    
    print("Training CNN Model...")
    cnn_model, cnn_history = dlm.train_cnn_model(cnn_data, cnn_labels)
    
    # Collect and display results
    print("Collecting and displaying results...")
    collect_and_display_results(pomt_results, gco_results, rnn_model, cnn_model, rnn_history, cnn_history, rnn_data, rnn_labels, cnn_data, cnn_labels)

def collect_and_display_results(pomt_results, gco_results, rnn_model, cnn_model, rnn_history, cnn_history, rnn_data, rnn_labels, cnn_data, cnn_labels):
    print("\n=== PoMT Consensus Results ===")
    print(f"Average Delay: {pomt_results['avg_delay']:.4f} seconds")
    print(f"Total Delay: {pomt_results['total_delay']:.4f} seconds")
    print(f"Blockchain Length: {len(pomt_results['blockchain'])}")
    print(f"Packet Delivery Ratio (PDR): {pomt_results['pdr']:.2f}%")
    
    print("\n=== GA Sidechain Optimization Results ===")
    print(f"Fitness: {gco_results['fitness']}")
    print(f"Energy Needed: {gco_results['energy_needed']:.4f} mJ")
    print(f"Number of Sidechains: {len(gco_results['sidechains'])}")
    
    print("\n=== RNN Model Summary ===")
    rnn_model.summary()
    
    print("\n=== CNN Model Summary ===")
    cnn_model.summary()
    
    # Visualization of training history
    plot_training_history(rnn_history, 'RNN Training History')
    plot_training_history(cnn_history, 'CNN Training History')
    
    # Calculate and display Precision, Recall, and F1 Score
    print("\n=== RNN Model Metrics ===")
    rnn_predictions = (rnn_model.predict(rnn_data) > 0.5).astype("int32")
    print(f"Precision: {precision_score(rnn_labels, rnn_predictions):.4f}")
    print(f"Recall: {recall_score(rnn_labels, rnn_predictions):.4f}")
    print(f"F1 Score: {f1_score(rnn_labels, rnn_predictions):.4f}")
    
    print("\n=== CNN Model Metrics ===")
    cnn_predictions = (cnn_model.predict(cnn_data) > 0.5).astype("int32")
    print(f"Precision: {precision_score(cnn_labels, cnn_predictions):.4f}")
    print(f"Recall: {recall_score(cnn_labels, cnn_predictions):.4f}")
    print(f"F1 Score: {f1_score(cnn_labels, cnn_predictions):.4f}")

def plot_training_history(history, title):
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{title} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{title} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
