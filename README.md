# Multi-sensor labelled multi-Bernoulli filters

This repository contains Matlab implementations of various labelled multi-Bernoulli (LMB) and LMB mixture (LMBM) filters.
The repository contains both single- and multi-sensor implementations of the LMB and LMBM filters.
These filters can be implemented using a variety of data association algorithms; however, all filters assume the linear-Gaussian dynamics.
The code is hopefully documented well enough that it is easy to interpret.
You should be able to use Matlab's **help** function to access the scripts' documentation.
This code runs in Matlab R2022a and requires the **Statistics and Machine Learning Toolbox**, but only to simulate Poisson distributed random variables.

## Quick Start

**Single-sensor:**
```matlab
setPath;
model = generateModel(10, 0.95, 'LBP', 'Fixed');
[gt, meas, gtRfs] = generateGroundTruth(model);
est = runLmbFilter(model, meas);
plotResults(model, meas, gt, est, gtRfs);
```

**Multi-sensor:**
```matlab
setPath;
model = generateMultisensorModel(3, [5 5 5], [0.67 0.70 0.73], [4 3 2], 'PU', 'LBP', 'Fixed');
[gt, meas, gtRfs] = generateMultisensorGroundTruth(model);
est = runParallelUpdateLmbFilter(model, meas);
plotMultisensorResults(model, meas, gt, est, gtRfs);
```

## Algorithms

Tracks multiple objects under clutter, missed detections, and unknown cardinality using linear-Gaussian dynamics. Supports LMB (fast, approximate) and LMBM (exact, expensive) filters with three data association methods:

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| **LBP** (Recommended) | Fast (~100-200ms) | High marginal approximation | Real-time, large problems |
| **Gibbs Sampling** | Moderate (~200-500ms) | MCMC convergence | LMBM filters, exact events |
| **Murty's Algorithm** | Moderate (~150-400ms) | Exact K-best | Deterministic associations |

**Multi-sensor fusion:** PU-LMB (best accuracy, assumes independent sensors), GA-LMB (best localization, poor cardinality), AA-LMB (balanced), IC-LMB (sequential processing).

## Key Structures

**Model:** `generateModel(clutterRate, detectionProb, 'LBP', 'Fixed')` - Contains state/measurement dimensions, transition matrices (A, R, C, Q), detection/survival probabilities, birth parameters, data association method.

**Measurements:** `{t}` = 2×M matrix (single-sensor) or `{s,t}` (multi-sensor) of [x; y] positions (mix of true detections + clutter).

**State Estimates:** `.labels{t}` (object IDs), `.mu{t}` ([x; y; vx; vy] states), `.Sigma{t}` (4×4 covariances).

**Ground Truth:** `.state{t}` per object, `.startTime/.endTime`, RFS format with `.cardinality(t)`.

## Filters

**Single-sensor** (`runFilters.m`): LMB supports all data association methods (LBP recommended). LMBM requires Gibbs/Murty (LBP incompatible). Configure in `common/generateModel.m`.

**Multi-sensor** (`runMultisensorFilters.m`): PU/GA/AA-LMB (parallel update variants), IC-LMB (iterated-corrector), Multi-sensor LMBM (exact, memory-intensive). Configure in `common/generateMultisensorModel.m`.

**Benchmarks:** `trials/` - OSPA metrics and runtime comparisons; `marginalEvaluations/` - data association algorithm comparisons.

## Octave Compatibility

Compatible with GNU Octave 9.0+ (1.5-2x slower than MATLAB).

**Setup:** Install statistics package, compile MEX file for Gibbs/Murty:
```bash
octave --eval "pkg install -forge statistics"
cd common/ && cp assignmentoptimal.c assignmentoptimal.cc && mkoctfile --mex assignmentoptimal.cc
```

**Usage:** Add `pkg load statistics;` before running scripts. LBP works without MEX compilation.

**Errors:** `assignmentoptimal undefined` → compile MEX; `poissrnd undefined` → load statistics package.

## Single-sensor LMB and LMBM filters

The script **runFilters.m** runs the single-sensor LMB and LMBM filters and plots their results and Euclidean and Hellinger optimal subpattern assignment (OSPA) metrics.
The LMB filter can be run using the following three data association algorithms:

   1. Loopy belief propagation (LBP). This is Williams et al.'s LBP algorithm that approximates each object's posterior existence probability and marginal association probabilities. We recommend this data association algorithm, as it is computationally inexpensive and it is more accurate than the other two data association algorithms.
   2. Gibbs sampling. This uses a relatively inexpensive Gibbs sampling routine to approximate each object's posterior existence probability and marginal association probabilities.
   3. Murty's algorithm. This uses Vo and Vo's **.mex** implementation of Murty's algorithm to approximate each object's posterior existence probability and marginal association probabilities.

All of the single-sensor LMB filters approximate each object's spatial distribution using a Gaussian mixture (GM), and the filters' parameters can be set in the script **common/generateModel.m**.
The LMBM filter can be implemented using both the Gibbs sampler and Murty's algorithm.
However, it cannot be implemented using the LBP algorithm, as that algorithm cannot be used to generate data association events and can only approximate marginal distributions.

## Multi-sensor LMB and LMBM filters

The script **runMultisensorFilters.m** runs the multi-sensor LMB and LMBM filters and plots their results and Euclidean and Hellinger OSPA metrics.
We have developed the following three approximate multi-sensor LMB filters:

  1. The parallel update LBM (PU-LMB) filter. This filter results from the mathematical manipulation of the multi-sensor multi-object Bayes filter's posterior distribution. This filter assumes that the sensors are independent, and it is the most accurate of our approximate multi-sensor LMB filters.
  2. The geometric average LMB (GA-LMB) filter. This filter is based on geometric average (GA) fusion, and it approximates the multi-sensor multi-object Bayes filter's posterior distribution using the weighted GA of each sensor's measurement-updated distribution. This filter is accurate in terms of object localisation (i.e. the objects' kinematic states), but provides a poor instantaneous cardinality (number of objects) estimate. However, it usually does track all the objects present. This filter does not assume the sensors are independent, and it provides a poor covariance estimate for independent sensors.
  3. The arithmetic average LMB (AA-LMB) filter. This filter is based on arithmetic average (AA) fusion, and it approximates the multi-sensor multi-object Bayes filter's posterior distribution using the weighted AA of each sensor's measurement-updated distribution. This filter does not assume the sensors are independent, and it provides the worst results of all our filters for independent sensors. However, its cardinality estimate is superior to the GA-LMB filter's.

The three multi-sensor LMB filters listed above all split the multi-sensor measurement update into independent single-sensor measurement updates, before combining the resulting measurement-updated distributions together.
Their measurement updates can be computed in parallel significantly reducing their computational cost.
However, in these implementations, the filters' measurements updates are not computed in parallel, and the **Parallel Computing Toolbox** is not required.
Only the AA-LMB filter propagates GMs, the PU- and GA-LMB filters assume an object's prior spatial distribution is Gaussian and approximates each object's posterior spatial distribution as Gaussian.
It is also possible to implement all three filters using a Gibbs sampler or Murty's algorithm.

We have also implemented an iterated-corrector LMB (IC-LMB) filter that computes each sensor's measurement update in sucession.
The IC-LMB filter propagates a GM for each object's spatial distribution, and it represents a typical implementation of a multi-sensor LMB filter.
We have also implemented a multi-sensor LMBM filter using a variant of our Gibbs sampler.
This filter represents an exact closed-form solution to the multi-sensor multi-object Bayes filter's recursion.
It accounts for a variable number of sensors; however, it is prohibitively expensive.
If you track a large number of objects using many sensors, then you will exceed Matlab's memory limit.

All of the multi-sensor filters' parameters are set in the script **common/generateMultisensorModel.m**.

## Other things

We also have some additional scripts that allow us to compare and contrast both our data association algorithms and filters.
The scripts are organised into the following two folders:

   1. **marginalEvaluations/:** The scripts in this folder compare the LBP data association's approximate marginal distrubtions to those produced by Murty's algorithm and our Gibbs sampler. 
The Gibbs sampler is based on the same underlying model as the LBP algorithm.
   2. **trials/:** The scripts in these folders compare the various single- and multi-sensor filters' OSPA metrics and runtimes in various scenarios. 
