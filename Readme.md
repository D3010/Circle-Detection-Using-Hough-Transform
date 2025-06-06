# Hough Transform for Circle Detection

A robust implementation of the Hough Transform for detecting circles in images. This repository contains a complete pipeline—from pre-processing and edge detection to accumulating votes and thresholding—to accurately detect circular shapes in input images.

---

## Table of Contents

- [Hough Transform for Circle Detection](#hough-transform-for-circle-detection)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Architecture](#architecture)
  - [Detailed Code Breakdown](#detailed-code-breakdown)

---

## Introduction

The Hough Transform is a classical technique for detecting geometric shapes within images by converting points in image space into curves in a parameter space. In this project, the goal is to detect circles given a known radius by estimating the circle centers. The implementation handles:

- Edge detection (using the Canny algorithm)
- Convolution-based filtering
- Voting in an accumulator array for circle center determination
- Visualization of both the detected circle centers and the vote distribution (accumulator)

This project serves both as an educational tool and as a robust solution for circle detection in practical computer vision applications.

---

## Architecture

The implementation is divided into two primary aspects:

1. **Theoretical Background:**
   - **Parameter Space:** A circle is characterized by its center coordinates (a, b) and a fixed radius (r).
   - **Accumulator Array:** A 2D array collects votes for candidate centers. Each edge pixel votes for all possible centers that could result in a circle of radius r.
   - **Quantization:** The angle (θ) is discretized (typically between 0° and 360° with a fixed step size) to transform the continuous parameter space into discrete bins.
   - **Thresholding:** A dynamic threshold (based on a ratio of the maximum vote count) filters out weak candidate centers.

2. **Practical Implementation:**
   - **Preprocessing and Edge Detection:** A grayscale image is processed using Canny edge detection.
   - **Convolution Operation:** Filtering techniques are applied using a custom 2D convolution function.
   - **Voting Process:** The algorithm votes for potential circle centers based on the computed angles.
   - **Visualization:** Displays the results with overlays on the original image and heatmaps of the accumulator.

---
## Installation
  - git clone https://github.com/yourusername/hough-transform-circle-detection.git
  - cd hough-transform-circle-detection
  - pip install -r requirements.txt
