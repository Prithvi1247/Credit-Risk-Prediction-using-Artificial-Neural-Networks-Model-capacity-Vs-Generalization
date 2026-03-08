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
- [lower capacity ->> slower memorization]

### Metrics 
    accuracy  ≈ 0.79
    precision ≈ 0.64
    recall    ≈ 0.64
    F1        ≈ 0.644

    recall = precision -> model is balanced

## LARGE ARCHITECTURE MODEL (60 → 128 → 64 → 32 → 1)
    18000 parameters
- Extremely More degrees of freedom than data complexity requires.
- The neural network begins to overfit after approximately 6 epochs, as indicated by increasing validation loss despite decreasing training loss.

- model checkpoint is around epoch 6.
- After that = training improves , generalization worsens
- [Higer capacity ->> faster memorization]

### Metrics 
    accuracy  ≈ 0.785
    precision ≈ 0.633
    recall    ≈ 0.644
    F1        ≈ 0.638

    recall = precision -> model is balanced

## EXPERIMENTAL INSIGHTS

1. 
    small model -> slower overfitting
    medium model -> balanced performance
    large model -> fastest memorization

    because

    larger networks have more parameters
        - higher capacity
        - easier memorization

2. 
    Even though the large model had the lowest validation loss (~0.487), its final metrics are slightly worse.

    Because the model: overfits extremely fast

### UNDERSTANDING

Increasing neural network capacity leads to faster memorization of the training data. While the larger model achieves lower training loss, its validation performance deteriorates earlier, indicating overfitting. This suggests that moderate-capacity networks provide better generalization on small tabular datasets.


| Model            | Architecture | Best Epoch | Accuracy | Precision | Recall |  F1   |
| ---------------- | ------------ | ---------- | -------- | --------- | ------ | ---   |
| Small            | [32]         | 13         | 0.8      | 0.661     | 0.661  | 0.661 |
| Medium           | [64,32]      | 9          | 0.77     | 0.603     | 0.644  | 0.623 |
| Large            | [128,64,32]  | 6          | 0.775    | 0.616     | 0.627  | 0.622 |
| Medium + Dropout | [64,32]      | 12         | 0.785    | 0.633     | 0.644  | 0.639 |
