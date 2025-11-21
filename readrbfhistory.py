# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 10:03:25 2025

@author: Divya
"""
import pandas as pd
import matplotlib.pyplot as plt

# === 1. Read the CSV file ===
	filename = "drift_history.csv"   # change if needed
	df = pd.read_csv(filename)
	
	# === 2. Extract columns ===
	batch = df["Batch"]
	drift = df["Overall_Drift"]
	
	# === 3. Plot drift curve ===
	plt.figure(figsize=(8, 5))
	plt.plot(batch, drift, marker='o', color='purple', linewidth=2, label='Overall Drift')
	
	# Optional: add a smooth line if drift fluctuates heavily
	# from scipy.ndimage import gaussian_filter1d
	# drift_smooth = gaussian_filter1d(drift, sigma=1)
	# plt.plot(batch, drift_smooth, '--', color='black', label='Smoothed Drift')
	
	plt.xlabel("Batch Index")
	plt.ylabel("RBF Drift (1 - Similarity)")
	plt.title("Overall RBF Drift Across Batches")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.show()
