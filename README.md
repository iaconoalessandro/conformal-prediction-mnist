# Conformal Prediction on MNIST, in Base R

A replication of the two conformal prediction procedures described in Angelopoulos and
Bates (2021): a neural network trained from scratch in base R, then wrapped in a
calibration layer that converts its softmax outputs into prediction sets carrying a
distribution-free coverage guarantee.

Academic project — Statistics for Data Science, MSc in Data Science and Business
Informatics, University of Pisa, a.a. 2025/2026. Coursework, done in a pair with Irene
Mungo. The assignment was to reproduce part of a published paper in R; the paper is
included at `docs/Paper_to_replicate.pdf`.

## The idea

A classifier that says "this digit is an 8, 87% confident" gives you a number with no
guarantee attached. Conformal prediction converts that into a set — "{3, 8}" — chosen so
that the true label falls inside it at least 90% of the time, without assuming anything
about the model or the data distribution beyond exchangeability. The guarantee comes from
a held-out calibration set rather than from the model.

Two ways of building the set are implemented:

- **LAC** (least ambiguous set-valued classifier): score each calibration point by
  `1 - p(true class)`, take the appropriate quantile, and include every class whose
  probability clears the resulting threshold.
- **APS** (adaptive prediction sets): sort the classes by probability and accumulate
  until the cumulative mass reaches the true class. Sets are built by adding classes in
  order until the same threshold is crossed.

## Data

MNIST, loaded through the `dslabs` package. 8,000 images are sampled from the combined
60,000 training and 10,000 test images, then split 50/25/25 into 4,000 training, 2,000
calibration and 2,000 test. Pixel values are scaled to [0, 1] — without that, the
gradient descent in the network diverges. Labels are one-hot encoded for the
cross-entropy loss.

Nothing about the data is messy; MNIST is clean by construction. The constraint that
shapes the project is size: 4,000 training images and a single hidden layer, which caps
accuracy well below what MNIST allows and directly determines how large the prediction
sets come out.

## Approach

**The network**, implemented in base R with no ML libraries: 784 inputs, one hidden layer
of 64 units with ReLU, a 10-way softmax output. Weights initialised from N(0, 0.01²),
full-batch gradient descent at learning rate 0.1 for 400 epochs, with the forward pass
and the backward pass both written by hand. Final training loss 0.3096.

**Calibration.** The conformal quantile is the `ceiling((n+1)(1-alpha))`-th smallest
calibration score, with alpha = 0.10 for a 90% target. LAC and APS each get their own
score function and their own threshold.

**Validation.** Beyond the single split, the calibration and test sets are pooled and
re-split 50 times at random, and coverage and set size are recomputed each time. This is
what turns a single lucky number into an estimate with a spread. Coverage is also swept
across 20 values of alpha from 0.01 to 0.39, and broken down per digit.

## What did not work

**APS massively over-covers, and this is the main negative result.** Against a 90% target
it delivers 99.57% coverage averaged over 50 splits, with prediction sets containing 4.20
of the 10 possible digits on average. A set that lists four digits out of ten has barely
narrowed anything down. Over-coverage is not a violation of the guarantee — the guarantee
is a lower bound and APS clears it — but it makes the method useless in practice here.

The cause is in the implementation: the APS score is the cumulative probability mass up
to and including the true class, with no randomised tie-breaking term. The randomised
variant in the source paper exists precisely to remove this conservatism, and it was not
implemented. With a poorly separated softmax the effect compounds, because the threshold
lands at 0.996 and reaching that much cumulative mass takes many classes.

**LAC produces empty prediction sets.** 11 of 2,000 test images, or 0.55%, come back with
no class at all, because no class clears the threshold. An empty set is consistent with
marginal coverage — it simply counts as a miss — but it is not an answer, and per-instance
it is worse than useless.

**Marginal coverage is not per-class coverage, and the gap is wide.** LAC hits 89.81%
overall, close to target. Broken down by digit, coverage runs from 81.11% on digit 5 to
97.41% on digit 1. The guarantee is marginal by construction and makes no per-class
promise, and these numbers show how far that drifts in practice. Digit 5 is also the
hardest for the network to classify at 76.11% accuracy, confused mainly with 3 and 8, so
the classes that most need a reliable set are the ones that get the least reliable one.

**The committed console log does not match the committed script.** `images/console_output.txt`
is 259 lines and contains per-digit breakdowns, set-size distributions, score statistics,
reliability-diagram data and a test-accuracy figure. The script in `code/` prints 32
lines and contains 12 `cat()` calls; it computes all of those quantities and plots them,
but never prints them. The log was produced by a more instrumented version of the script
that is not in this repository. The numbers in it are consistent with what the script
does compute, but a reader who runs the code cannot currently reproduce most of the
tables below from the console.

## Results

Single split, 2,000 test images:

| | LAC | APS |
| :--- | ---: | ---: |
| Threshold (qhat) | 0.689 | 0.996 |
| Empirical coverage | 91.45% | 99.60% |
| Mean set size | 1.07 | 4.24 |
| Median set size | 1 | 4 |
| Set size range | 0 to 2 | 1 to 10 |
| Empty sets | 11 (0.55%) | 0 |

Averaged over 50 random calibration/test re-splits, target coverage 90%:

| Method | Mean coverage | Std dev | Range | Mean set size |
| :--- | ---: | ---: | :--- | ---: |
| LAC | 89.81% | 0.91% | 87.55 – 92.25% | 1.03 |
| APS | 99.57% | 0.11% | 99.35 – 99.85% | 4.20 |

LAC lands within 0.2 points of its 90% target, which is what the theory predicts and is
the result the project set out to verify. APS overshoots by nearly 10 points.

The underlying network reaches 91.83% training accuracy and 88.70% test accuracy. Per
digit, test accuracy ranges from 76.11% on digit 5 to 95.81% on digit 0.

Coverage tracks the target closely across the alpha sweep for LAC — at targets of 0.99,
0.95, 0.90 and 0.85 the empirical values are 0.9925, 0.954, 0.9225 and 0.8395 — while
average set size falls from 2.95 to 0.90 across the same range. That trade-off, tighter
coverage for larger sets, is the practical content of the method.

Comparing the two methods image by image: APS returns a strictly larger set than LAC on
1,813 of 2,000 images, an equal-sized set on 187, and a smaller one on none.

## Reproducing

```bash
git clone https://github.com/iaconoalessandro/conformal-prediction-mnist.git
cd conformal-prediction-mnist
Rscript code/requirements.R          # installs dslabs, the only dependency
Rscript code/01_Conformal_Prediction_MNIST.R
```

The script downloads MNIST through `dslabs` on first run, so it needs network access
once. Everything else — the network, backpropagation, both conformal procedures, and the
50-split evaluation — is base R. Runtime is a few minutes on CPU. Plots go to the default
device, which under `Rscript` means an `Rplots.pdf` written to the working directory; the
versions committed under `images/` were exported separately.

Two caveats on reproducing, both verified against R 4.6.1 with dslabs 0.9.1:

- **The script prints far less than `images/console_output.txt` shows.** See the last
  point under *What did not work*. The per-digit tables, set-size distributions and score
  statistics are computed and plotted but not printed.
- **`set.seed(42)` does not make the run bit-identical across environments.** On R 4.6.1
  the run gives 91.47% training accuracy, 89.96% average LAC coverage and 99.67% average
  APS coverage, against 91.83%, 89.81% and 99.57% in the committed log — differences of
  around 0.3 percentage points, from the sample of 8,000 images landing differently.
  Every conclusion above is unaffected, but the digits will not match exactly.

## Tech stack

| Purpose | What is used |
| :--- | :--- |
| Language | base R (4.x) |
| Data | `dslabs`, for MNIST only |
| Neural network | written from scratch — forward pass, backpropagation, softmax, ReLU |
| Conformal prediction | written from scratch — LAC and APS scores, quantile, set construction |
| Plots | base R graphics |

`dslabs` is the single external dependency, and it is used only to fetch the dataset.

## Repository layout

```
code/
  01_Conformal_Prediction_MNIST.R   the whole project, in one script
  requirements.R                    installs dslabs
docs/
  Paper_to_replicate.pdf            Angelopoulos & Bates (2021)
  Presentation.pdf / .pptx          project slides as delivered
  Project_Guidelines.pdf            the course assignment brief
images/                             figures, plus console_output.txt
```

## Reference

Angelopoulos, A. N., and Bates, S. (2021). *A Gentle Introduction to Conformal Prediction
and Distribution-Free Uncertainty Quantification.* Included at
`docs/Paper_to_replicate.pdf`.

## Authors

Alessandro Iacono and Irene Mungo.
