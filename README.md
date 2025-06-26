# 🧠 Neural Network from Scratch – Step-by-Step ANN Implementation in Python

Welcome to the **Neural Network From Scratch** project! 🚀
This repository provides a **comprehensive, beginner-friendly** guide to building an **Artificial Neural Network (ANN)** using **pure Python** — no high-level libraries like TensorFlow or PyTorch. This is a perfect learning tool if you're diving into machine learning or want to deeply understand how ANNs work under the hood.

---

## 📌 What You'll Learn

* How artificial neurons are implemented mathematically and programmatically
* How forward and backward propagation works in neural networks
* How loss functions and accuracy are calculated
* How weights are updated using gradient descent
* How to visualize decision boundaries
* The performance of a simple ANN on a 3-class spiral dataset

---

## 🧪 How to Run the Code

### 🔧 Prerequisites

Make sure you have Python 3.x installed along with the following libraries:

```bash
pip install numpy matplotlib
```

### 🚀 Run the Full Neural Network

To train and test the neural network on the spiral dataset:

```bash
python 00_full_ann_final.py
```

This will:

* Generate a synthetic dataset (3-class spiral)
* Train the neural network using softmax + categorical cross-entropy
* Plot training loss and accuracy
* Display the decision boundary after training

---

## 📈 Performance Metrics

Below is the result after training the ANN:

* 📉 **Loss**: Decreased from \~1.098 to \~1.083
* 📈 **Accuracy**: Improved gradually to **\~43%**
* 📊 **Decision Boundary**: Shows rough separation of the 3 classes, but performance is still basic due to the simplicity of the network

--- 

COMPLETE NEURAL NETWORK FROM SCRATCH

1. Generating spiral dataset...
   Dataset shape: (300, 2)
   Labels shape: (300,)
   Classes: [0 1 2]

2. Building neural network...
   Architecture:
   Input (2) -> Dense(3) -> ReLU -> Dense(3) -> Softmax -> CrossEntropy

3. Training for 53000 epochs...

   Learning rate: 0.01

   Epoch     0 | Loss: 1.0986 | Accuracy: 0.3400
   
   Epoch  1000 | Loss: 1.0986 | Accuracy: 0.3767
   
   Epoch  2000 | Loss: 1.0986 | Accuracy: 0.3467

   Epoch  3000 | Loss: 1.0986 | Accuracy: 0.3700

   Epoch  4000 | Loss: 1.0986 | Accuracy: 0.3733

   Epoch  5000 | Loss: 1.0986 | Accuracy: 0.3733

   Epoch  6000 | Loss: 1.0986 | Accuracy: 0.3733

   Epoch  7000 | Loss: 1.0985 | Accuracy: 0.3800

   Epoch  8000 | Loss: 1.0985 | Accuracy: 0.3767

   Epoch  9000 | Loss: 1.0984 | Accuracy: 0.3733

   Epoch 10000 | Loss: 1.0982 | Accuracy: 0.3833

   Epoch 11000 | Loss: 1.0979 | Accuracy: 0.3700

   Epoch 12000 | Loss: 1.0973 | Accuracy: 0.3733

   Epoch 13000 | Loss: 1.0965 | Accuracy: 0.3667

   Epoch 14000 | Loss: 1.0952 | Accuracy: 0.3733

   Epoch 15000 | Loss: 1.0935 | Accuracy: 0.3800

   Epoch 16000 | Loss: 1.0915 | Accuracy: 0.3733

   Epoch 17000 | Loss: 1.0895 | Accuracy: 0.3733

   Epoch 18000 | Loss: 1.0878 | Accuracy: 0.3767

   Epoch 19000 | Loss: 1.0863 | Accuracy: 0.3800

   Epoch 20000 | Loss: 1.0852 | Accuracy: 0.3900

   Epoch 21000 | Loss: 1.0843 | Accuracy: 0.3967

   Epoch 22000 | Loss: 1.0840 | Accuracy: 0.4000

   Epoch 23000 | Loss: 1.0838 | Accuracy: 0.4067

   Epoch 24000 | Loss: 1.0837 | Accuracy: 0.4067

   Epoch 25000 | Loss: 1.0836 | Accuracy: 0.4067

   Epoch 26000 | Loss: 1.0836 | Accuracy: 0.4100

   Epoch 27000 | Loss: 1.0836 | Accuracy: 0.4100

   Epoch 28000 | Loss: 1.0836 | Accuracy: 0.4100

   Epoch 29000 | Loss: 1.0836 | Accuracy: 0.4100

   Epoch 30000 | Loss: 1.0836 | Accuracy: 0.4100

   Epoch 31000 | Loss: 1.0836 | Accuracy: 0.4100

   Epoch 32000 | Loss: 1.0836 | Accuracy: 0.4100

   Epoch 33000 | Loss: 1.0836 | Accuracy: 0.4067

   Epoch 34000 | Loss: 1.0836 | Accuracy: 0.4067

   Epoch 35000 | Loss: 1.0836 | Accuracy: 0.4067

   Epoch 36000 | Loss: 1.0836 | Accuracy: 0.4067

   Epoch 37000 | Loss: 1.0835 | Accuracy: 0.4033

   Epoch 38000 | Loss: 1.0835 | Accuracy: 0.4033

   Epoch 39000 | Loss: 1.0835 | Accuracy: 0.4033

   Epoch 40000 | Loss: 1.0835 | Accuracy: 0.4033

   Epoch 41000 | Loss: 1.0835 | Accuracy: 0.4033

   Epoch 42000 | Loss: 1.0835 | Accuracy: 0.4033

   Epoch 43000 | Loss: 1.0834 | Accuracy: 0.4000

   Epoch 44000 | Loss: 1.0834 | Accuracy: 0.4067

   Epoch 45000 | Loss: 1.0834 | Accuracy: 0.4067

   Epoch 46000 | Loss: 1.0833 | Accuracy: 0.4100

   Epoch 47000 | Loss: 1.0832 | Accuracy: 0.4167

   Epoch 48000 | Loss: 1.0832 | Accuracy: 0.4200

   Epoch 49000 | Loss: 1.0831 | Accuracy: 0.4267

   Epoch 50000 | Loss: 1.0830 | Accuracy: 0.4267

   Epoch 51000 | Loss: 1.0828 | Accuracy: 0.4200

   Epoch 52000 | Loss: 1.0827 | Accuracy: 0.4267

5. Training completed!
   
   Final Loss: 1.0826

   Final Accuracy: 0.4333

7. Sample predictions (first 5 samples):

   True labels: [0 0 0 0 0]

   Predictions: [1 1 1 1 1]

   Probabilities:

   Sample 0: [0.31497345 0.35855036 0.32647619]

   Sample 1: [0.31339976 0.3613713  0.32522895]

   Sample 2: [0.3126557 0.3629602 0.3243841]

   Sample 3: [0.31504138 0.35933326 0.32562536]

   Sample 4: [0.31446095 0.36061901 0.32492003]

9. Plotting training progress...

10. Visualization complete!

---

![NN From Scratch](https://github.com/user-attachments/assets/af7282cf-b8cd-4199-9caa-642f8629ada9)

---

## 💡 Key Highlights

* **Educational Focus**: Each file represents a clear step in the construction of an ANN.
* **From Scratch**: Everything — forward pass, backward pass, gradients — is manually implemented.
* **Visualization Included**: Training progress and decision boundaries are visualized for better understanding.
* **Modular Design**: Each component (layer, activation, loss, optimizer) is in a separate file for clarity.

---

## 👨‍🏫 Ideal For

* Students and self-learners who want to understand the internals of ANNs
* Developers transitioning into machine learning
* Anyone curious about how libraries like TensorFlow and PyTorch work under the hood

---

## 📬 Contributing

Feel free to fork this repo, improve the model, or refactor the code. PRs are always welcome!

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

---

Let me know if you want to:

* Add explanations in the code itself
* Extend it to support hidden layers or different optimizers
* Improve the performance further

I'm happy to help you take this further!

---

## 🙏 Acknowledgments

This project was inspired and guided by the outstanding work of **Harrison Kinsley**, widely known as [**Sentdex**](https://www.youtube.com/@sentdex).

His YouTube series "**Neural Networks from Scratch**" helped me deeply understand how ANNs function at the lowest level, and it was the foundation for this project.

📺 YouTube Series: [Neural Networks from Scratch – Sentdex](https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)  
🐙 GitHub: [github.com/Sentdex](https://github.com/Sentdex)  
🐦 Twitter: [@sentdex](https://twitter.com/sentdex)

> Thank you, Sentdex, for your generous contribution to the learning community!


