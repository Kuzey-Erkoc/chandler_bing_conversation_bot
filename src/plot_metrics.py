import os
import matplotlib.pyplot as plt
from conversation import output_dir
 
def plot_training_loss(losses):
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(losses)+1), losses, marker="o")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()
 