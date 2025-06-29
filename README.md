# Day 2: Building a Neural Network from Scratch

## Topic: Feedforward Neural Networks (FNNs)

---

### Summary:

* Understanding **Forward Propagation** in multi-layer networks
* Exploring **Loss Functions** (MSE, Cross-Entropy)
* Intuition behind **Backpropagation** and **Gradient Descent**
* The **training loop**
* **Implementing** a simple FNN in TensorFlow with real data

---

## 1. What is a Feedforward Neural Network?

A **Feedforward Neural Network** (FNN) is the simplest form of a neural network where information moves in only **one direction** — **forward** from input to output.

###  Analogy: Water Flowing Through Pipes

> Imagine a system of pipes. Water flows from the top (input) through valves (neurons) and exits at the bottom (output). It **never flows backward** — that's feedforward.

---

## 2. Forward Propagation in Multi-layer Networks

### Structure:

* **Input Layer:** Receives raw features (e.g., pixels, text embeddings)
* **Hidden Layers:** Learn representations
* **Output Layer:** Makes predictions

### Forward Pass:

1. Each neuron computes a weighted sum:
   $z = w_1x_1 + w_2x_2 + \ldots + b$
2. Applies an activation function:
   $a = \text{activation}(z)$
3. Passes the result to the next layer

###  Analogy: Assembly Line

> Think of an assembly line where raw materials (inputs) are processed at each station (layer), transformed, and passed to the next station until the final product (prediction) is ready.

---

## 3. Loss Functions

Loss functions measure how far the model's predictions are from the actual values.

### 3.1 Mean Squared Error (MSE)

Used in **regression tasks**.

$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

###  Analogy:

> Imagine throwing darts at a bullseye. MSE is the **average squared distance** of all your darts from the center. The closer they are, the lower the loss.

### 3.2 Cross-Entropy Loss

Used in **classification tasks**.

$L = -\sum y \log(\hat{y})$

###  Analogy:

> Think of it like a **lie detector**: it punishes the model more when it's confidently wrong than when it's unsure.

---

## 4. Backpropagation (Intuition Only)

Backpropagation is the process of updating weights by **propagating the error backwards** from the output to each neuron.

### Steps:

1. Compute loss
2. Calculate gradient (how change in weight affects loss)
3. Update weights using gradient descent

###  Analogy: Cooking Feedback Loop

> Imagine you're cooking a dish. If it tastes too salty, you reduce salt next time. You trace back the mistake to the **ingredient (weight)** that caused the bad result (loss).

---

## 5. Gradient Descent (Basic Intuition)

It’s an optimization algorithm that helps us **minimize the loss**.

$\theta = \theta - \alpha \nabla L(\theta)$
Where:

* $\theta$ = weight parameters
* $\alpha$ = learning rate
* $\nabla L$ = gradient of loss

### Analogy: Mountain Descent

> You’re standing in fog on a hill. You take small steps **downhill** (in the direction of steepest descent) until you reach the lowest point (minimum loss).

---

## 6. Training Loop Overview

A training loop typically looks like this:

1. Pass data through the network (**forward pass**)
2. Compute loss
3. Backpropagate errors
4. Update weights
5. Repeat for many epochs

###  Analogy:

> Like teaching a child. You show an example, they guess, you correct them, they improve — over many repetitions (epochs).

---

## 7. Implementing a Simple Neural Network in TensorFlow

We'll use the **Fashion MNIST** dataset — classifying items like shirts, shoes, bags, etc.

```python
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

# Load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer=SGD(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
```

### Explanation:

* `Flatten`: Converts 28x28 image into a 1D array
* `Dense`: Fully connected layer
* `ReLU`: Activation for hidden layers
* `Softmax`: Outputs probabilities for 10 classes
* `SGD`: Optimizer using gradient descent

---

## 8. Applications of Feedforward Neural Networks

| Domain    | Use Case                              |
| --------- | ------------------------------------- |
| Health    | Predict disease risk from symptoms    |
| Finance   | Classify transactions as fraud or not |
| Retail    | Recommend products                    |
| Education | Predict student drop-out likelihood   |

---

## 9. Real-World Applications of Deep Learning

### Healthcare

* Disease diagnosis from X-rays and MRIs
* Predicting patient deterioration in hospitals
* Drug discovery and genomics

### Finance

* Credit scoring and risk assessment
* Fraud detection in real-time
* Algorithmic trading and portfolio optimization

### Transportation

* Autonomous vehicles (e.g., Tesla Autopilot)
* Traffic prediction and route optimization
* Predictive maintenance for vehicles

### Retail & E-commerce

* Personalized product recommendations (e.g., Amazon)
* Dynamic pricing models
* Visual search (e.g., find clothes based on a picture)

### Entertainment

* Content recommendation (Netflix, Spotify)
* Deepfake generation and detection
* Auto-captioning and video summarization

### Agriculture

* Disease detection in crops via image classification
* Monitoring soil conditions with sensor data
* Yield prediction

### Cybersecurity

* Detecting anomalies in network traffic
* Classifying malware using byte-level data
* Real-time phishing detection

### Legal & Document Processing

* Automating contract review
* Classifying and extracting entities from legal text
* Predicting case outcomes

---

##  Final Analogy Recap

| Concept          | Analogy                        |
| ---------------- | ------------------------------ |
| Feedforward      | Water through pipes            |
| Loss Function    | Dart distance or lie penalty   |
| Backpropagation  | Cooking feedback loop          |
| Gradient Descent | Walking downhill in fog        |
| Training Loop    | Teaching a child by correction |

---

