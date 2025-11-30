# Hippocampal SNN Crypt (v0.1 PoC)
**Bio-inspired Data Compression & Encryption based on Spiking Neural Networks**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

## 🧠 Overview
This project is a Proof of Concept (PoC) for a next-generation data compression and encryption algorithm inspired by the information processing mechanism of the **Hippocampal Dentate Gyrus** (Pattern Separation & Sparse Coding).

Unlike traditional algorithms (ZIP, AES), this system converts data into **"Spike Patterns"** of a neural network. The synaptic weights (structure of the network) act as the "Private Key". Without the exact neural structure, reconstructing the original data from the spike patterns is mathematically infeasible, achieving **simultaneous compression and encryption**.

> **Based on academic research**
> Inspired by the author's graduate student research on "input interactions in the dentate gyrus of the hippocampus."

## 🚀 Features
*   **Bio-inspired Security**: Based on physiological models of the rat hippocampus (Granule Cells).
*   **Structure as Key**: The randomly generated neural network weights act as the decryption key.
*   **Sparse Coding**: Drastically reduces data representation by recording only the indices of firing neurons (Winners-Take-All).

## 📦 Usage

1. Requirements
pip install numpy pillow

2. Workflow
Step 1: Generate the Brain (Key)
Create a random neural network structure. This file (snn_key.pkl) is required for both encryption and decryption.
python 01_generate_snn.py
(Tested successfully with a 32x32 black-and-white image in v0.1)

Step 2: Compress (Encrypt)
Place your target image in the input_data/ directory.
Run the compression script to generate spike data.
python 02_compress.py
Output: output_data/compressed.dat

Step 3: Decompress (Decrypt)
Reconstruct the image from the spike data using the pre-generated key.
python 03_decompress.py
Output: output_data/restored_image.png

## 📅 Roadmap

v0.1: Proof of Concept (Single image sparse coding)

v0.5: Support for arbitrary file types and folder batch processing

v1.0: Auto-optimization of neuron count based on data size

v2.0: Evolutionary algorithm (GA) to search for the most efficient SNN structure

## 👤 Author
hafufu-stack
note: https://note.com/cell_activation/m/m5bf070b82882