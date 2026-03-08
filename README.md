Credit Risk Prediction using Artificial Neural Networks

Input: financial information about a person
Output: probability of loan default

binary classification problem.

0 → safe borrower
1 → risky borrower

Dataset used: German Credit Dataset

Goal: 
    demonstrate understanding of neural networks
    run experiments
    analyze results

Neural Network Modeling for Credit Risk Prediction: 
An Experimental Study on Tabular Financial Data


Train multiple ANN architectures
Compare their behavior
Analyze overfitting and generalization

# DATASET : 
    1000 SAMPLES
    After train/test split:
        ~800 training samples
    

## MEDIUM ARCHITECTURE MODEL (60 → 64 → 32 → 1)
    6000 parameters
- more degrees of freedom than data complexity requires.
- The neural network begins to overfit after approximately 9 epochs, as indicated by increasing validation loss despite decreasing training loss.

- model checkpoint is around epoch 9–10.
- After that = training improves , generalization worsens

### Metrics 
    accuracy  ≈ 0.795
    precision ≈ 0.65
    recall    ≈ 0.66
    F1        ≈ 0.655

    recall = precision -> model is balanced

## SMALL ARCHITECTURE MODEL (60 → 32 → 1)
    1920 parameters
- Still more degrees of freedom than data complexity requires.
- The neural network begins to overfit after approximately 12-13 epochs, as indicated by increasing validation loss despite decreasing training loss.

- model checkpoint is around epoch 12-13.
- After that = training improves , generalization worsens

### Metrics 
    accuracy  ≈ 0.79
    precision ≈ 0.64
    recall    ≈ 0.64
    F1        ≈ 0.644

    recall = precision -> model is balanced

