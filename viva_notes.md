# EEEM068 LSA Viva Notes

## 1. Research problem

### Why is melanoma classification challenging?

- Severe class imbalance.
- False-negative melanoma predictions are important.
- Multiple images may originate from the same patient.
- Clinical metadata may provide information complementary to images.

---

## 2. Validation

### Why use patient-aware splitting?

Patient-aware splitting prevents observations belonging to the
same patient from appearing in both training and validation data.

This reduces the risk of information leakage and provides a more
realistic estimate of performance on unseen patients.

### What is patient leakage?

Patient leakage occurs when images belonging to the same
patient appear in both training and validation data.

A model may then exploit patient-specific similarities,
making validation performance appear better than its
performance on genuinely unseen patients.

### How was it prevented?

I used StratifiedGroupKFold.

- `target` was used for stratification.
- `patient_id` was used as the grouping variable.
- Five folds were generated.
- Fold 0 was used as the fixed validation set.
- The same split was used for every model.
- Known patient overlap between training and validation
  was explicitly verified to be zero.

### Did I perform full five-fold cross-validation?

No.

Five patient-aware folds were generated to define a
reproducible grouped partition, but only Fold 0 was used
as the held-out validation set for the LSA experiments.

This was done to maintain a common validation set across
experiments while keeping training computationally feasible.
---

## 3. Logistic Regression


### Logistic Regression

Logistic Regression is a linear probabilistic classifier.
It models the log-odds of the positive class as a weighted
linear combination of input features and maps the result
through the logistic sigmoid function.

In H0/H1, the input was a 4,096-dimensional vector created
by converting each lesion image to grayscale, resizing it
to 64x64, scaling pixel values to [0,1], and flattening it.

The main limitation is that flattening removes explicit
spatial structure. Logistic Regression does not learn
hierarchical image features in the way a CNN does.

### Class weighting

H1 used class_weight="balanced".

This gives minority-class observations greater influence
during model fitting. It changes the optimisation objective;
it is not the same as simply changing the classification
threshold afterward.

### Decision threshold

The predicted probability is converted into a class using
a threshold.

Lowering the threshold generally makes the classifier more
willing to predict melanoma, which can increase sensitivity
while also increasing false positives and reducing
specificity.

### ROC-AUC

ROC-AUC measures how well the model ranks positive cases
above negative cases across thresholds.

It is threshold-independent.

### Precision-recall / Average Precision

Precision-recall analysis is particularly useful for this
dataset because melanoma represents only about 1.76% of
observations.

### Why use the same Fold 0 for every model?

A common validation set means performance differences can
be attributed more meaningfully to model/method changes
rather than different validation samples.

---

## 4. CNNs

TODO

---

## 5. ResNet18

TODO

---

## 6. ConvNeXt

TODO

---

## 7. BCEWithLogitsLoss

TODO

---

## 8. Weighted BCE

TODO

---

## 9. Evaluation metrics

TODO

---

## 10. Reusable PyTorch pipeline

### Dataset and DataLoader

A Dataset maps one index to one sample, including the image,
target, image name, and patient ID. A DataLoader wraps the
Dataset to create batches and control shuffling and worker-based
loading. Training is shuffled, while validation preserves a
deterministic order.

### Why is validation not augmented?

Validation uses only deterministic resizing and normalization.
Stochastic augmentation would change the evaluation samples
between passes and make model comparisons less reproducible.

### BCEWithLogitsLoss

The model produces an unbounded binary logit.
`BCEWithLogitsLoss` combines the sigmoid operation and binary
cross-entropy in a numerically stable implementation.

### Positive-class weighting

For weighted BCE, `pos_weight` is calculated from the training
partition as the number of negative examples divided by the
number of positive examples. This gives positive melanoma
examples greater contribution to the training loss without
using validation labels.

### Transfer learning and fine-tuning

B0 starts from ResNet18 weights pretrained on ImageNet and
replaces the 1,000-class classifier with one binary output.
The complete network will be fine-tuned rather than freezing the
pretrained backbone.

### Automatic mixed precision

On CUDA, autocast can use lower-precision operations where
appropriate, while gradient scaling reduces the risk of very
small gradients underflowing. This can reduce memory use and
computation time, but it does not guarantee identical results
across every platform.

### Why is sigmoid not part of the model?

The loss expects raw logits, so adding sigmoid inside the model
would duplicate that operation and reduce numerical stability.
Sigmoid is applied only when probabilities are needed during
validation or prediction.
