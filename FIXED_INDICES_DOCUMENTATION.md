# Fixed Indices Data Split Strategy

##  **Rigorous Data Separation**

This package uses **fixed indices** to ensure consistent, leak-free data splits across all experiments.

##  **The 4-Split System**

### 1. **Training Set** (~516 samples)
- Used for training each probe model
- Varies by probe but consistent indices

### 2. **Validation Set** (~64 samples) 
- Used for hyperparameter tuning and early stopping
- Consistent across all probes

### 3. **Test Set** (33 samples)
```python
GLOBAL_TEST_LIST = [484, 22, 633, 594, 43, 623, 325, 248, 573, 543, 513, 515, 328, 20, 629, 496, 208, 286, 96, 420, 474, 270, 645, 280, 346, 460, 266, 104, 300, 129, 352, 145, 146]
```

### 4. **True Validation Set** (33 samples)
```python
GLOBAL_TRUE_VAL_LIST = [2, 171, 55, 506, 544, 407, 333, 46, 612, 371, 399, 546, 186, 422, 440, 419, 275, 249, 176, 250, 398, 3, 461, 362, 503, 175, 306, 521, 117, 550, 198, 447, 598]
```

## ✅ **Data Integrity Guaranteed**

- **Zero overlap** between test and true_val sets
- **Same samples** always in same splits across all 867 probe models
- **Combined test set**: 66 samples for robust evaluation
- **No data leakage** possible

##  **Implementation in Package**

```python
# The package automatically uses these fixed indices
from ensemble_ploidy_classifier.utils.model_loader import PretrainedModelLoader

loader = PretrainedModelLoader()
rankings = loader.get_model_rankings()  # Based on fixed test set performance
```

##  **For New Species**

When adapting to new species:

1. Define your own fixed indices for your dataset size
2. Maintain the same split proportions
3. Ensure no overlap between splits
4. Document your indices for reproducibility

