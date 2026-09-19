# Human Activity Recognition using Six HMMs

## Overview

This project implements Human Activity Recognition (HAR) using six
activity-specific Hidden Markov Models (HMMs).

The UCI Human Activity Recognition Using Smartphones Dataset is used
for training and testing.

Instead of using one HMM for all activities, a separate HMM is trained
for each of the six activities.

## Activities

The six activities are:

1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Lying Down

## Dataset

The project uses the UCI HAR Dataset.

- 30 subjects
- Smartphone inertial sensor data
- Accelerometer and gyroscope measurements
- 50 Hz sampling frequency
- 561 extracted features per sample
- 7352 training samples
- 2947 testing samples

The dataset is already divided into training and testing sets.

## Method

The implementation follows this pipeline:

```text
UCI HAR Dataset
       |
       v
Load Training Data
       |
       v
Create Subject/Activity Sequences
       |
       v
Split Data by Activity
       |
       v
+-----------------------------+
|      Six Activity HMMs      |
+-----------------------------+
| Walking                     |
| Walking Upstairs            |
| Walking Downstairs          |
| Sitting                     |
| Standing                    |
| Lying Down                  |
+-----------------------------+
       |
       v
Save Trained Models
       |
       v
Evaluate on Test Data