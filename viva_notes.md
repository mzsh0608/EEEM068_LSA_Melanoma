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

CNNs learn local image patterns with shared filters and build increasingly
abstract spatial features across layers. This preserves image structure,
unlike flattening grayscale pixels for Logistic Regression.

---

## 5. ResNet18

### Why ResNet18?

ResNet18 is a relatively compact, established CNN that supports ImageNet
transfer learning and is computationally practical for a controlled baseline.
It tests whether learned hierarchical RGB features improve over the flattened
grayscale H0/H1 representation without introducing a large architecture.

### What is a residual connection?

A residual block learns a transformation that is added to its input through a
skip connection. This gives gradients a direct route through the network and
helps deeper networks optimise without requiring every block to relearn an
identity mapping.

### Transfer learning and full fine-tuning

B0 started from `IMAGENET1K_V1` weights, replaced the 1,000-class classifier
with one binary logit, and updated the complete network. Pretraining supplies
general visual features; full fine-tuning adapts both those features and the
new classifier to lesion images.

---

## 6. ConvNeXt

### What is ConvNeXt?

ConvNeXt is a convolutional architecture that modernises a residual CNN using
design choices such as a patch-like stem, large-kernel depthwise convolution,
LayerNorm, GELU, pointwise channel expansion, and stochastic depth. It remains
a hierarchical image model and retains residual connections.

### How does it differ conceptually from ResNet18?

ResNet18 mainly uses standard 3x3 convolutions with BatchNorm and ReLU inside
residual blocks. Torchvision ConvNeXt uses 7x7 depthwise spatial convolution,
separate pointwise channel mixing, LayerNorm, GELU, layer scaling, and drop
path. Many internal properties change together, so M1 compares architecture
families rather than isolating one component.

### What are depthwise convolution, LayerNorm, and GELU?

A depthwise convolution applies one spatial filter per input channel, leaving
channel mixing to later pointwise layers. LayerNorm normalises features within
each sample rather than using batch statistics. GELU is a smooth nonlinear
activation that scales values according to their magnitude instead of applying
the hard zero boundary used by ReLU.

### What is stochastic depth or drop path?

During training, stochastic depth can randomly bypass a residual branch for
some samples. This regularises the network while the full deterministic model
is used for evaluation. M1 retained torchvision ConvNeXt-Tiny's built-in
stochastic-depth design and did not add custom dropout.

### Why ConvNeXt-Tiny?

Tiny is the smallest standard ConvNeXt variant and was practical on the
available GPU while still providing a stronger-capacity comparison with
ResNet18. Its binary model had 27,820,897 parameters, versus 11,177,025 for
B0, so larger ConvNeXt variants would increase an already substantial compute
difference.

### Why keep B0 preprocessing and optimisation choices?

M1 used the same Fold 0, 224x224 RGB inputs, transforms, weighted BCE,
training-only `pos_weight`, AdamW, learning rate, weight decay, AMP policy,
epoch budget, early stopping, and threshold as B0. Matching these external
choices reduces confounding, so the principal controlled change is the image
architecture. Individually tuning each architecture could improve its result,
but would weaken this architecture comparison and increase computational cost.

### Why transfer learning and full fine-tuning?

M1 started from official ImageNet `IMAGENET1K_V1` weights, replaced only the
final classifier with one logit, and updated all parameters. Pretraining
provides general visual features; full fine-tuning adapts the entire hierarchy
to lesion images.

---

## 7. BCEWithLogitsLoss

### Why one logit and no sigmoid layer?

Binary classification needs one score: a positive logit favours melanoma and
a negative logit favours benign. `BCEWithLogitsLoss` combines sigmoid and
binary cross-entropy in a numerically stable calculation, so the model must
provide raw logits. Sigmoid is applied later when evaluation probabilities are
needed.

### Forward and backward passes

The forward pass maps a batch of images to logits and computes the weighted
loss against its labels. Backpropagation calculates the gradient of that loss
for every trainable parameter, after which AdamW updates the parameters.

---

## 8. Weighted BCE

### Why weight the positive class?

Only 467 of 26,499 training examples are melanoma. B0 and M1 used
`pos_weight = 26032 / 467 = 55.74304068522484`, so errors on positive training
examples contribute more strongly to the objective. This changes optimisation;
it is not threshold tuning or oversampling.

### Why is Fold 0 excluded from weighting?

`pos_weight` describes the training distribution and affects gradient updates.
Using Fold 0 labels in it would allow validation information to influence
fitting. The 6,627 Fold 0 samples were used only for epoch validation,
checkpoint selection, and final evaluation.

---

## 9. Evaluation metrics and B0/M1 evidence

### Why AdamW and AMP?

AdamW provides adaptive parameter updates while applying decoupled weight
decay; B0 used learning rate and weight decay of 0.0001. On CUDA, AMP used
autocast and gradient scaling to reduce computation and memory demand while
protecting numerical stability.

### Why checkpoint by ROC-AUC?

ROC-AUC assesses ranking across thresholds and is less dominated by the large
benign majority than raw accuracy. The criterion was declared before training.
B0 therefore restored epoch 4, which had the highest validation ROC-AUC
(0.864276), even though Average Precision happened to peak at epoch 6.

### Why is accuracy misleading here?

H0 achieved 0.9810 accuracy while detecting only 1 of 117 melanomas. B0 had
lower accuracy (0.7681) but detected 93 melanomas, giving sensitivity 0.7949
and balanced accuracy 0.7812. Accuracy alone rewards majority-class
predictions in this imbalanced dataset.

### What happened during B0 training?

Training loss generally declined from 1.0615 to 0.8120. Validation ROC-AUC
peaked at epoch 4 and then fell over epochs 5-7 while validation loss rose.
Early stopping triggered after three non-improving epochs and restored epoch
4. This pattern is consistent with overfitting or instability, but the
fluctuations were not tested for statistical significance.

### What did B0 show compared with H0/H1?

B0 improved ROC-AUC to 0.8643 from 0.6570 (H0) and 0.6244 (H1), and Average
Precision to 0.1193 from 0.0386 and 0.0363. At threshold 0.5 it found 93
melanomas, versus 1 and 17, showing that the pretrained CNN did not collapse
to near-all-benign predictions. The cost was 1,513 false positives,
specificity of 0.7676, and precision of only 0.0579.

### What limitations remained?

B0 was evaluated on one fixed fold, without confidence intervals or external
validation. Weighted BCE improved sensitivity but left a substantial
sensitivity-specificity trade-off and low precision. No threshold was tuned,
so the reported operating point remains the common predeclared threshold 0.5.

### What happened during M1 training?

M1's ROC-AUC and Average Precision both peaked at epoch 4 (0.9008 and
0.1694). Training loss generally decreased through epoch 6 but rose at epoch
7. Validation loss spiked at epoch 5 and then recovered, while post-peak
ROC-AUC remained close to 0.892. Early stopping triggered after epoch 7 and
restored epoch 4. Both B0 and M1 showed fluctuation, but M1's later ROC-AUC and
validation loss were more stable than B0's stronger post-peak deterioration.

### What changed from B0 to M1?

The image architecture and its internal parameter count changed: ResNet18 had
11,177,025 trainable parameters and ConvNeXt-Tiny had 27,820,897. The data,
Fold 0, transforms, image size, loss, `pos_weight`, optimiser settings, AMP,
epoch budget, early stopping, checkpoint metric, and threshold remained the
same. M1 took 767.5 seconds versus 593.8 seconds for B0.

### Which model performed better?

Under the fixed protocol, M1 improved ROC-AUC from 0.8643 to 0.9008, Average
Precision from 0.1193 to 0.1694, balanced accuracy from 0.7812 to 0.8010, and
sensitivity from 0.7949 to 0.9744. It detected 114 melanomas rather than 93
and reduced false negatives from 24 to 3.

The threshold trade-off worsened in other respects: M1 produced 2,424 false
positives rather than 1,513, specificity fell from 0.7676 to 0.6276,
precision fell from 0.0579 to 0.0449, and F1 fell from 0.1080 to 0.0859.
Therefore M1 ranked cases better and detected more melanomas, but it did not
produce a uniformly better threshold-0.5 classifier.

### What are the limitations of the architecture comparison?

Only one fixed fold was evaluated, without confidence intervals or external
validation. The architectures differ in many internal mechanisms and
parameter count, so no single component can be credited causally. The common
hyperparameters favour experimental control but may not be individually
optimal. The results do not establish clinical utility or statistically
significant architecture superiority.

---

## 10. Metadata fusion (M2)

### What is metadata fusion?

Metadata fusion combines the learned image representation with structured
patient or lesion attributes before classification. M2 concatenated a
768-dimensional ConvNeXt image embedding with a 32-dimensional metadata
embedding and mapped the result to one raw logit.

### Why use age, sex, and anatomical site?

These variables were available before diagnosis and may contain context that
is not fully visible in lesion pixels. Age can reflect differing disease
prevalence, sex may correlate with population-level risk patterns, and site
provides lesion-location context. Their use does not mean they cause the
prediction or that demographic differences are clinically acceptable.

### Why not use diagnosis?

Diagnosis directly describes the lesion outcome and can encode the target.
Using it as an input would leak answer-like information and invalidate the
classification experiment.

### What is target leakage?

Target leakage occurs when a predictive input contains information that would
not legitimately be available when making the intended prediction, especially
information derived from the outcome. It can produce misleadingly high
validation performance.

### Why fit preprocessing on training only?

Imputation, scaling, and category discovery all learn facts about a dataset.
Fitting them on Fold 0 would let validation values shape the model input space,
so M2 fitted once on 26,499 training rows and only transformed Fold 0.

### What are median imputation and the training median?

Median imputation replaces missing ages with the middle observed training age.
M2's training median was 50.0. The training median is used because validation
ages must not influence fitting and the median is less sensitive to extreme
values than the mean.

### Why standardise age, and what does StandardScaler learn?

Standardisation places age on a scale more compatible with encoded categorical
features and neural optimisation. StandardScaler learned a training mean of
48.59164496773463 and scale of 14.20582128293888, then reused them unchanged
for Fold 0.

### Why one-hot encode sex and site?

These categories have no defensible numeric order. One-hot encoding gives each
training category its own indicator rather than implying that one location or
sex is numerically greater than another.

### What does handle_unknown="ignore" do?

An unseen validation category maps to zeros for that variable's learned
indicators instead of expanding the feature space or failing. No such
validation-only category occurred in the real Fold 0, but the behaviour is
covered by synthetic tests.

### Why not use patient_id as a predictive input?

Patient ID is an identity and grouping key, not a clinical predictor. Encoding
it could encourage memorisation and undermine evaluation on unseen patients;
M2 kept it only for splitting and prediction audits.

### What is an image embedding, and where was it extracted?

An image embedding is a compact learned feature vector representing the image.
M2 used the normal ConvNeXt features, adaptive pooling, classifier LayerNorm,
and flatten operations, taking the 768 values immediately before the original
final classification linear layer.

### What are feature concatenation and the metadata MLP?

Concatenation joins the 768 image features and 32 metadata features into one
800-element vector. The metadata MLP was a learned linear projection from the
11 preprocessed inputs to 32 values, followed by GELU and dropout.

### Why a small 32-dimensional embedding, GELU, and dropout?

Thirty-two dimensions provide limited nonlinear metadata capacity without
letting a small structured input branch dominate model size. GELU matches the
smooth activation family used by ConvNeXt. Dropout 0.20 regularises the new
branch during training and is disabled at evaluation.

### Why only one final fusion linear layer?

The deliberately small head keeps the ablation interpretable: the principal
addition is metadata information, not a large new classifier. It also added
only 416 parameters relative to M1.

### Why initialize M2 from ImageNet rather than M1?

Loading M1 would give M2 extra melanoma-supervised training before fusion and
confound the comparison. Both image encoders therefore started independently
from the same ImageNet V1 weights, while M2 jointly trained its image,
metadata, and fusion branches.

### What changed and what remained fixed from M1 to M2?

M2 added training-only metadata preprocessing, an 11-to-32 metadata branch,
concatenation, and a fusion classifier. The seed, ConvNeXt-Tiny family,
ImageNet initialization, train/Fold 0 rows, image pipeline, weighted BCE,
`pos_weight`, AdamW settings, AMP, training budget, checkpoint criterion, and
threshold remained fixed.

### Did metadata improve performance, and what trade-offs changed?

It did not improve ranking on this fold: ROC-AUC changed from 0.900848 to
0.897446 and AP from 0.169406 to 0.165292. At threshold 0.5, M2 reduced false
positives by 705 and improved specificity, precision, balanced accuracy, and
F1, but produced 9 more false negatives and reduced sensitivity from 0.974359
to 0.897436.

### What limitations does the M2 ablation have?

It uses one validation fold, no confidence intervals, no external validation,
and no calibration or threshold analysis. The three metadata variables were
added together, so the result cannot be attributed to age, sex, or site
individually without separate ablations. The evidence does not establish
statistical significance or clinical utility.

---

## 11. Reusable PyTorch pipeline

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
