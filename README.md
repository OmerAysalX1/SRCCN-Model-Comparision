# 🖼️ Deep Learning-Based Super-Resolution: SRCNN Model Comparison  

This repository contains the implementation and comparison of **Super-Resolution Convolutional Neural Networks (SRCNN)** under different configurations. The goal is to systematically evaluate how **input channel selection** (RGB vs. Y channel) and **deep learning framework** (TensorFlow vs. PyTorch) affect the performance of SRCNN in single image super-resolution tasks.  

---

## 📖 Table of Contents
- [🎯 Project Objective](#-project-objective)  
- [📊 Dataset & Preprocessing](#-dataset--preprocessing)  
- [🧠 Model Architecture](#-model-architecture)  
- [⚙️ Training Setup](#️-training-setup)  
- [📈 Results](#-results)  
- [🛠️ Technical Details](#️-technical-details)  
- [👥 Team](#-team)  
- [📌 Feedback & Contributions](#-feedback--contributions)  

---

## 🎯 Project Objective
The main objectives of this study are:  
- ✅ Evaluate **SRCNN trained on RGB channels** using TensorFlow.  
- ✅ Evaluate **SRCNN trained only on luminance channel (Y from YCbCr)** using TensorFlow.  
- ✅ Compare **two luminance-based SRCNNs** trained with different frameworks (**TensorFlow vs. PyTorch**).  

By analyzing these three scenarios, the project investigates the impact of:  
- Input data representation (RGB vs. Y channel).  
- Framework differences (TensorFlow vs. PyTorch).  
- Normalization techniques on model convergence and PSNR performance.  

---

## 📊 Dataset & Preprocessing
- **Dataset:** High-resolution (HR) images and their corresponding low-resolution (LR) versions obtained via **bicubic interpolation**.  
- **Preprocessing Steps:**  
  - For **RGB-based SRCNN**:  
    - Normalize each channel using channel-wise mean and standard deviation.  
  - For **Y-channel SRCNN**:  
    - Convert RGB → YCbCr.  
    - Use only the Y (luminance) channel.  
    - Normalize values to `[0,1]`.  
- **Patch Extraction:** HR and LR images are cropped into patches for training.  
- **File Formats:** `.png` images processed via **PIL** and **Torchvision transforms**.  

---

## 🧠 Model Architecture
All models follow the original **SRCNN** design from Dong et al. (2014):  

1. **Conv1:** 9×9 convolution, 64 filters, ReLU activation  
2. **Conv2:** 1×1 convolution, 32 filters, ReLU activation  
3. **Conv3:** 5×5 convolution, 1 filter, linear activation  

Key points:  
- Input: Either full RGB image (3 channels) or Y channel (1 channel).  
- Output: Super-resolved image of the same dimension as input.  
- Loss Function: **Mean Squared Error (MSE)**  

---

## ⚙️ Training Setup
- **Optimizer:** Adam  
- **Batch Size:** 16  
- **Epochs:** 10  
- **Learning Rate:**  
  - First two layers: `1e-4`  
  - Last layer: `1e-5`  
- **Frameworks:** TensorFlow 2.x & PyTorch 2.x  
- **Hardware:** GPU acceleration with CUDA (if available)  

---

## 📈 Results

| Model ID | Input Type | Framework   | Avg. PSNR (dB) | Final Loss |
|----------|-----------|-------------|----------------|------------|
| Model-1  | RGB       | TensorFlow  | **30.37**      | 0.0160     |
| Model-1  | Y channel | TensorFlow  | 27.99          | 0.0118     |
| Model-2  | Y channel | PyTorch     | 24.48          | 0.003761   |

### 🔍 Observations
- 🎨 **RGB model (TensorFlow):** Best performance with **30.37 dB PSNR**, showing the importance of full color information.  
- 🌌 **Y channel model (TensorFlow):** Retains structural details (27.99 dB PSNR) but loses color information.  
- ⚡ **Y channel model (PyTorch):** Lowest PSNR (**24.48 dB**) → highlights differences in **normalization techniques and framework implementations**.

### 🖼️ Model Visualizations
<img width="2816" height="1556" alt="image" src="https://github.com/user-attachments/assets/392c42e2-7f25-4bbe-9956-b8eb47dc9785" />
<img width="2810" height="1582" alt="image" src="https://github.com/user-attachments/assets/c5a2863e-ea6f-422d-a12f-337fdb4ccfdf" />
<img width="2808" height="1588" alt="image" src="https://github.com/user-attachments/assets/1734fac6-6b99-45a3-a836-e17d07f87e0c" />

- Currently, there are no sample images available in the `dataset/` folder.
- For the remaining images or the full dataset, please **contact me**; files can be shared upon request.

### 📂 Dataset Note
- All HR/LR training and test images are stored in the `dataset/` folder.
- Due to size restrictions, **no images are included in the repository**.
- The full dataset is available upon request to collaborators or interested researchers.
---

## 🛠️ Technical Details
- **Programming Language:** Python 3.10+  
- **Deep Learning Libraries:**  
  - TensorFlow 2.14 (Keras API)  
  - PyTorch 2.1 + Torchvision  
- **Image Processing:** PIL, NumPy  
- **Evaluation Metrics:**  
  - **MSE (Mean Squared Error)**  
  - **PSNR (Peak Signal-to-Noise Ratio)**  

---

## 👥 Team
- **Ömer Aysal**  
- **Melisa Aslan**   

---

## 📌 Feedback & Contributions
We welcome feedback and contributions 🚀  
- Open an **Issue** for feature requests or bug reports.  
- Submit a **Pull Request** for improvements or experiments.  
- Any suggestions regarding **model design, preprocessing, or evaluation metrics** are highly appreciated.  

---

