# Model Performance Summary

| Model Type           | Dataset Type | Features | Epochs | Test Accuracy (%) | Training Time | Description |
|---------------------|-------------|---------|--------|-----------------|---------------|-------------|
| Quantum NN (VQC)    | Grayscale   | 4       | 20     | 39.79           | ~4h           | Only 4 features due to hardware limitations and training time |
| Quantum NN (VQC)    | Binary      | 4       | 20     | 21.47           | ~4h           | Only 4 features due to hardware limitations and training time |
| Classical NN        | Grayscale   | 4       | 20     | 78.36           | ~10s          | Small feature set for comparison with quantum models |
| Classical NN        | Binary      | 4       | 20     | 66.96           | ~10s          | Small feature set for comparison with quantum models |
| Classical NN        | Grayscale   | 44      | 20     | 96.73           | ~10s          | Features retained 95% variance using PCA |
| Classical NN        | Binary      | 55      | 20     | 88.66           | ~10s          | Features retained 95% variance using PCA |


## Conclusion

The models were trained and tested on 19,702 images. Images were first preprocessed to 8×8 grayscale and binary format, then flattened into feature vectors, converted to tensors, and finally fed into the models. Classical neural networks outperform the current quantum models in both accuracy and speed. For classical models, PCA was used to retain 95% of the variance, resulting in 44 features for grayscale and 55 for binary, which significantly improved performance. Quantum models used only 4 features due to hardware and simulation limitations, restricting their expressiveness and leading to lower accuracy and slower training.

