# 🤚 Hand Language Classification with CNN

## 📘 Overview
This project focuses on classifying **hand gestures from images** using **Convolutional Neural Networks (CNNs)** with **data augmentation**.  

Initially, models were trained **without preprocessing**, resulting in poor performance. Traditional techniques such as **HOG, DCT, and DWT** improved results moderately, but the **CNN with augmentation far outperformed them**, achieving **nearly 100% accuracy on both training and test sets**.

The project demonstrates how **deep learning can automatically extract relevant features and handle complex image classification tasks**, far surpassing traditional feature-based methods.

---

## ⚙️ Features
✅ Classify 25 different hand gestures from images  
✅ **Data augmentation** to simulate variations in rotation, scale, and position, improving generalization  
✅ Automatic **feature extraction** with CNN layers  
✅ Prevent overfitting using **dropout** and **batch normalization**  
✅ Real-time evaluation on training and validation datasets  

---

## 🧠 How It Works

The **CNN** automatically extracts hierarchical features from images.  

1. **Data Augmentation:**  
   - The images are randomly rotated, zoomed, and shifted to simulate real-world variations in hand gestures.  
   - This increases the diversity of the training data, allowing the network to **generalize better** and reduce overfitting.  

2. **Convolutional Layers:**  
   - These layers slide learnable kernels over the image, performing element-wise multiplication and summation to detect **local features** like edges and textures.  
   - Activation functions, such as **ReLU**, introduce non-linearity, allowing the network to learn **complex patterns**.  

3. **Pooling Layers:**  
   - **Max-pooling** reduces spatial dimensions while preserving the most important features, making the network **more efficient and robust**.  

4. **Fully Connected Layers:**  
   - After feature extraction, the resulting feature maps are flattened into a vector and passed through dense layers.  
   - **Dropout** randomly disables a fraction of neurons to prevent overfitting.  
   - The final layer outputs class probabilities for 25 gesture categories using a **softmax function**.  

5. **Training:**  
   - The network is trained using the **Adam optimizer**, which adapts learning rates and uses momentum to accelerate convergence.  
   - With augmented data, the network achieves **nearly perfect accuracy** on both training and test sets.  

**Key Takeaways:**  
- CNNs automatically learn the most relevant features, making manual preprocessing mostly unnecessary.  
- Data augmentation improves the model’s ability to generalize to unseen gestures.  
- The combination of convolution, pooling, dropout, and batch normalization ensures **robust performance** with minimal overfitting.  

---

## 🧩 Tech Stack

| Library | Purpose |
|---------|---------|
| **Python** | Programming language |
| **TensorFlow / Keras** | CNN implementation and training |
| **NumPy** | Numerical computations |
| **skimage** | Image preprocessing |
| **Matplotlib / Seaborn** | Visualization |
| **Pandas** | data manipulation and analysis

---
