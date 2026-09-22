# Human Activity Recognition using Six HMMs
# One-HMM Human Activity Recognition

## Overview

This module implements a **single Hidden Markov Model (HMM)** for Human Activity Recognition (HAR) using the **UCI Human Activity Recognition Using Smartphones Dataset**.

The goal is to train one Gaussian HMM on the extracted smartphone sensor features and use the model to predict hidden activity states for the test data.

## Dataset

The project uses the UCI HAR Dataset, which contains smartphone sensor measurements collected from subjects performing six activities:

1. WALKING
2. WALKING_UPSTAIRS
3. WALKING_DOWNSTAIRS
4. SITTING
5. STANDING
6. LAYING

The dataset contains **561 feature measurements** for each sample.

### Dataset Split

- Training samples: 7,352
- Testing samples: 2,947
- Features: 561

The dataset is stored under:

```text
One_HMM/Data/UCI HAR Dataset/
