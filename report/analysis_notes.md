# Phase I Analysis Notes

## Threshold behaviour

The fixed, predeclared grid was 0.10 to 0.90 in steps of 0.10. No threshold
was optimized on Fold 0, and 0.50 remains the primary model-comparison
threshold. The 0.50 rows exactly reproduced the frozen confusion matrices:
M1 had TN=4086, FP=2424, FN=3, TP=114; M2 had TN=4791, FP=1719, FN=12,
TP=105.

M1 sensitivity decreased from 1.000 at thresholds 0.10-0.30 to 0.974 at
0.50, 0.949 at 0.60, and 0.385 at 0.90. M2 sensitivity decreased from 0.957
at 0.10-0.20 to 0.897 at 0.50, 0.863 at 0.60, and 0.299 at 0.90. M2 was
more conservative at every grid point: it predicted fewer positive cases,
had higher specificity, and had lower sensitivity than M1 at each equal
threshold.

The trade-off is not a uniform model ordering. At similar high sensitivity,
M1 could retain more specificity (M1 at 0.60: sensitivity 0.949, specificity
0.695; M2 at 0.20: sensitivity 0.957, specificity 0.595). At similar
specificity, M2 could retain more sensitivity (M2 at 0.70: sensitivity 0.803,
specificity 0.851; M1 at 0.80: sensitivity 0.761, specificity 0.861).
Precision increased as thresholds rose. M1 F1 increased through 0.90, while
M2 F1 was highest on the inspected grid at 0.80 and changed little at 0.90;
these are descriptive observations, not selected operating points.

M1's slightly higher ROC-AUC and Average Precision summarize ranking over
thresholds, whereas the 0.50 metrics reflect one operating point and the
models' different score distributions. Therefore those results are not
contradictory.

## M2 high-confidence failures

The deterministic manifest contains six content-unique cases from each M2
confusion category. False negatives were ranked by ascending M2 melanoma
probability; false positives and true positives by descending probability;
true negatives by ascending probability. Byte-identical content was retained
at most once per category. All 24 original JPEGs were visually inspected.

Among six false negatives, visible hair occurred in five and low contrast in
four; one diffuse lesion reached the frame edge. `ISIC_0149568` and
`ISIC_9022005` are byte-identical validation JPEGs from the same patient.
`ISIC_9022005` was therefore replaced in the qualitative sample by the next
ranked unique-content FN, `ISIC_4967586`; both dataset rows remain in all
quantitative evaluation. Among six false
positives, hair was visible in three, ruler/ink markers in three, uneven colour
or illumination in two, and low contrast in two. One false positive was partly
out of focus and did not show a clearly delineated lesion. The six true
positives also included hair in three, ruler/marker artifacts in three, low
contrast in two, and one out-of-focus image. All six selected true negatives
contained visible hair, and four were low contrast.

Artifacts therefore occurred in correct as well as incorrect predictions.
These 24 deliberately extreme cases are a qualitative diagnostic sample; they
do not establish that any visible property caused a prediction.

## M1/M2 disagreements

Across all 6,627 validation images, prediction transitions at threshold 0.50
were: 3,968 M1-negative to M2-negative, 121 M1-negative to M2-positive, 835
M1-positive to M2-negative, and 1,703 M1-positive to M2-positive. The mean M2
minus M1 probability change was -0.072599 and the median was 0.000731.

The net reduction of 705 false positives is explained by 826 M1 FP to M2 TN
changes offset by 121 M1 TN to M2 FP changes. The nine additional false
negatives are exactly nine M1 TP to M2 FN changes; there were no M1 FN to M2 TP
changes. M1 and M2 were trained independently from ImageNet initialization, so
these paired prediction changes cannot be treated as the causal effect of the
metadata values.

## Subgroup behaviour

Sex used the observed female and male categories. Age bands were `<40`,
`40-59`, and `60+`, with boundaries age <40, 40 <= age <60, and age >=60.
There were no missing validation ages. Anatomical-site values were retained as
head/neck, lower extremity, oral/genital, palms/soles, torso, upper extremity,
and unknown. Every table reports N and positive count; groups with fewer than
10 positives are flagged as small, a reporting caution rather than a
significance boundary.

M2 specificity was higher than M1 specificity in every reported subgroup, but
M2 sensitivity was lower for both sexes and every age band. The female
sensitivity change was -0.109 (46 positives) and the male change was -0.056
(71 positives). The largest age-band sensitivity reduction was observed for
`<40` (-0.176; 17 positives), while the `60+` reduction was -0.017 (59
positives). Site sensitivity was unchanged for head/neck and lower extremity,
but fell by 0.093 for torso and 0.174 for upper extremity. The oral/genital,
palms/soles, and unknown-site groups had only 3, 1, and 1 positive cases,
respectively, so their AUC, AP, and sensitivity values are highly unstable.

The fraction of samples that changed binary prediction was 13.1% for female
and 15.8% for male cases, and 19.0%, 15.5%, and 10.0% across the `<40`,
`40-59`, and `60+` bands. Oral/genital and palms/soles had high change rates,
but their N and malignant counts were very small. These are descriptive
within-fold differences. They neither establish fairness nor prove demographic
bias or a causal metadata effect.

## Grad-CAM

Grad-CAM used the frozen best M2 checkpoint and the saved Phase H metadata
preprocessor without refitting it. Exact transformed metadata was supplied on
every forward pass. The dynamically verified target module was `features.7.2`,
the final spatial ConvNeXt block before pooling. Its activation and gradient
shapes were `[1, 768, 7, 7]`. Gradients targeted the raw melanoma logit; channel
weights were global averages of those gradients, followed by a weighted sum,
ReLU, normalization, and resizing to 512 x 512.

Three cases from each confusion category were reviewed. In the first two false
negatives, attribution was concentrated on the visible lesion despite the low
predicted probability; one also showed a strong bottom-right boundary hotspot.
The hair-obscured false negative had split attribution over the diffuse lesion
and a left-edge region. False-positive maps were broadly concentrated over the
central heterogeneous or erythematous fields; the ink-marked examples did not
show attribution exclusively on ink. The true-positive maps generally covered
the visible lesion, with separate lobes over pigmented regions in one case.
For the selected true negatives, positive-logit attribution was often weak on
the central lesion and stronger on peripheral, hair-covered, or edge regions.
Because every CAM targets the melanoma logit, these TN maps show regions that
contribute toward the positive logit despite a low overall score; they do not
directly explain why the model classified an image as benign.

All 12 CAMs were finite, normalized, spatially nontrivial, and reproduced the
saved M2 probabilities within tolerance. Hooks were closed after generation.
Grad-CAM is a coarse 7 x 7 attribution of the image branch conditioned on the
provided metadata. It does not explain the metadata branch, prove causal feature
use, or demonstrate dermatologist-like reasoning.

## Bootstrap uncertainty

A paired patient-level cluster bootstrap sampled validation patient IDs with
replacement and included all images for each sampled patient. M1 and M2 used
the same sampled patients. With seed 42, all 1,000 requested iterations were
valid and none were skipped.

The observed M2-M1 ROC-AUC difference was -0.003402; the bootstrap median was
-0.003234 and the 95% percentile interval was [-0.020408, 0.013015]. The
observed AP difference was -0.004114; the median was -0.006695 and the interval
was [-0.051608, 0.030815]. Both intervals span zero, so the observed ranking
differences are small relative to this bootstrap variability. The sensitivity
difference at 0.50 was -0.076923; its median was -0.075269 and interval was
[-0.139547, -0.026070]. These descriptive internal-validation intervals are not
a significance test or external validation.
