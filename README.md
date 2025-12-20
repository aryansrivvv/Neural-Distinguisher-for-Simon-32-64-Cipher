# Neural-Distinguisher-for-Simon-32-64-Cipher

The project is based on the research paper **"A Multi-Differential Approach to Enhance Related-Key Neural Distinguishers"** by Yuan and Wang, exploring how Deep Learning can be used to distinguish cryptographic outputs from random noise using Residual Networks (ResNet) and Squeeze-and-Excitation (SE) attention mechanisms.

## Project Overview
Cryptanalysis is the study of analyzing information systems to study the hidden aspects of the systems. This project bridges **Classical Cryptanalysis** (Differential Attacks) with **Deep Learning**.

The goal is to train a neural network to distinguish between:

1. **Related-Key Pairs:** Ciphertexts encrypted with keys and plaintexts satisfying a specific differential characteristic ( delta_in = delta_key).
2. **Random Pairs:** Ciphertexts generated from random data.

If the network achieves accuracy significantly > 50%, the cipher is considered "distinguished" for that number of rounds.

## Repository Structure
* `simon.py`: A pure Python/Numpy implementation of the **Simon 32/64** cipher, optimized for batch encryption.
* `data_generator.py`: Generates positive and negative training samples. Includes both **Single-Differential** and **Multi-Differential** generation logic.
* `model.py`: The Neural Network architecture (ResNet + Squeeze-and-Excitation blocks) implemented in PyTorch.
* `train.py`: The training loop, validation logic, and Cyclic Learning Rate scheduler.
* `README.md`: Project documentation.

## Methodology
1. The Cipher**Simon 32/64** is a Feistel-based lightweight cipher proposed by the NSA .

* **Block Size:** 32 bits
* **Key Size:** 64 bits
* **Operations:** Bitwise XOR, AND, Circular Shifts.

2. The Attack
We utilize a **Related-Key Differential Attack**. The paper posits that differentials where the Input Difference (\Delta P) equals the Key Difference (\Delta K) are particularly effective for neural distinguishers.


**Primary Differential:** `0x0000 0x0040` (Used for Single-Diff experiments).



**Secondary Differential:** `0x0000 0x8010` (Added for Multi-Diff experiments).



###3. The Model
The discriminator is a **Residual Network (ResNet)** enhanced with **Squeeze-and-Excitation (SE)** attention blocks.

* **Input:** 64-bit binary vector (concatenated ciphertext pair).
* **Hidden Layers:** 1D Convolutional layers with Residual connections.
  
**Attention:** SE blocks re-calibrate channel feature responses to focus on bit-level patterns .



## Installation & Usage
###Prerequisites
* Python 3.8+
* PyTorch
* Numpy

```bash
pip install torch numpy

```

###Running the TrainingTo train the distinguisher, run the `train.py` script. You can modify the `ROUNDS` variable inside the script to test different security margins.

```bash
python train.py

```

## Experimental Results - Experiments were conducted on a standard CPU/GPU setup using **raw ciphertext** (no partial decryption).

| Rounds | Method | Accuracy | Status |
| --- | --- | --- | --- |
| **6 Rounds** | Single-Differential | **83.18%** |  **Broken** |
| **7 Rounds** | Single-Differential | **60.50%** |  **Broken** |
| **7 Rounds** | Multi-Differential | **55.34%** |  Signal Dilution |
| **8 Rounds** | Single-Differential | **50.21%** |  Secure (Limit Reached) |

Key Findings
1. **Breaking 7 Rounds:** The Basic Neural Distinguisher successfully identified non-random patterns in the ciphertext up to 7 rounds with >60% accuracy.
2. **Signal Dilution:** At 7 rounds, the **Multi-Differential** approach yielded lower accuracy (55%) compared to the **Single-Differential** approach (60%). This suggests that mixing a strong differential (`0x40`) with a weaker one (`0x8010`) at this specific round count diluted the training signal, rather than enhancing it.

## References
1. **Yuan, X. & Wang, Q.** (2025). *A Multi-Differential Approach to Enhance Related-Key Neural Distinguishers*. The Computer Journal.
2. **Lu, J., et al.** (2024). *Improved (related-key) differential-based neural distinguishers for SIMON and SIMECK block ciphers*. The Computer Journal.
3. **Gohr, A.** (2019). *Improving attacks on round-reduced speck32/64 using deep learning*. CRYPTO 2019.

---

*This project was developed as part of CGE198-Cryptographic Standards and Cryptanalysis at Chennai Mathematical Institute*
