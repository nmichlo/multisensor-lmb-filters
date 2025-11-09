# Multi-sensor labelled multi-Bernoulli filters

This repository contains Matlab implementations of various labelled multi-Bernoulli (LMB) and LMB mixture (LMBM) filters.
The repository contains both single- and multi-sensor implementations of the LMB and LMBM filters.
These filters can be implemented using a variety of data association algorithms; however, all filters assume the linear-Gaussian dynamics.
The code is hopefully documented well enough that it is easy to interpret.
You should be able to use Matlab's **help** function to access the scripts' documentation.
This code runs in Matlab R2022a and requires the **Statistics and Machine Learning Toolbox**, but only to simulate Poisson distributed random variables.

## Table of Contents
- [Octave Compatibility](#octave-compatibility)
- [Architecture Overview](#architecture-overview)
- [Algorithm Basics](#algorithm-basics)
- [Inputs and Outputs](#inputs-and-outputs)
- [Single-sensor LMB and LMBM filters](#single-sensor-lmb-and-lmbm-filters)
- [Multi-sensor LMB and LMBM filters](#multi-sensor-lmb-and-lmbm-filters)
- [Other things](#other-things)

## Octave Compatibility

This code is also compatible with **GNU Octave** (tested with version 9.3.0) with minor setup required.

### Prerequisites

1. **Install Octave** (version 9.0 or later recommended)
2. **Install the statistics package:**
   ```bash
   octave --eval "pkg install -forge statistics"
   ```

### Setup Steps

#### 1. Compile the MEX file for Octave

The `assignmentoptimal` MEX file (used by Gibbs and Murty's algorithm) must be recompiled for Octave:

```bash
cd common/
cp assignmentoptimal.c assignmentoptimal.cc
mkoctfile --mex assignmentoptimal.cc
```

This creates `assignmentoptimal.mex` which Octave can use.

**Note:** This step is **required** if you want to use:
- Gibbs sampling data association
- Murty's algorithm data association
- LMBM filters (which depend on the above)

If you only use LBP (Loopy Belief Propagation), this compilation step is not necessary.

#### 2. Load the statistics package

Before running any scripts in Octave, load the statistics package:

```octave
pkg load statistics
```

Or add `pkg load statistics;` at the top of any script you run.

### Running with Octave

**Single-sensor example:**
```octave
pkg load statistics;
setPath;
model = generateModel(10, 0.95, 'LBP', 'Fixed');
[groundTruth, measurements, groundTruthRfs] = generateGroundTruth(model);
stateEstimates = runLmbFilter(model, measurements);
plotResults(model, measurements, groundTruth, stateEstimates, groundTruthRfs);
```

**Multi-sensor example:**
```octave
pkg load statistics;
setPath;
model = generateMultisensorModel(3, [5 5 5], [0.67 0.70 0.73], [4 3 2], 'PU', 'LBP', 'Fixed');
[groundTruth, measurements, groundTruthRfs] = generateMultisensorGroundTruth(model);
stateEstimates = runParallelUpdateLmbFilter(model, measurements);
plotMultisensorResults(model, measurements, groundTruth, stateEstimates, groundTruthRfs);
```

**Running trials:**
```bash
octave --no-gui trials/singleSensorAccuracyTrial.m
octave --no-gui trials/multiSensorAccuracyTrial.m
```

### Octave vs MATLAB Differences

- **Performance:** Octave is typically 1.5-2x slower than MATLAB for these filters
- **Plotting:** Some plot rendering may differ slightly
- **MEX files:** MATLAB and Octave MEX files are not cross-compatible and must be compiled separately
- **Compatibility:** All filters (LBP, Gibbs, Murty, LMB, LMBM, multi-sensor variants) work identically in Octave after MEX compilation

### Troubleshooting

**Error: `'assignmentoptimal' undefined`**
- Solution: Compile the MEX file using the steps in "Setup Steps" above

**Error: `'poissrnd' undefined`**
- Solution: Run `pkg load statistics` before executing scripts

**Error: path separators**
- The code uses Unix-style paths (`/`) which work on all platforms including Windows with both MATLAB and Octave

## Architecture Overview

### Directory Structure

```
.
├── common/                  # Shared utilities and core algorithms
│   ├── generateModel.m      # Single-sensor model configuration
│   ├── generateMultisensorModel.m  # Multi-sensor model configuration
│   ├── generateGroundTruth.m       # Synthetic trajectory generation
│   ├── loopyBeliefPropagation.m   # LBP data association
│   ├── murtysAlgorithm.m          # Murty's algorithm wrapper
│   ├── Hungarian.m                 # Hungarian algorithm for OSPA
│   ├── ospa.m                      # OSPA metric computation
│   └── assignmentoptimal.c/cc     # Optimal assignment (MEX)
├── lmb/                     # Single-sensor LMB filter
│   ├── runLmbFilter.m       # Main LMB filter entry point
│   ├── lmbPredictionStep.m  # Time prediction
│   └── lmbGibbsSampling.m   # Gibbs data association for LMB
├── lmbm/                    # Single-sensor LMBM filter
│   ├── runLmbmFilter.m      # Main LMBM filter entry point
│   └── lmbmPredictionStep.m # LMBM prediction with hypotheses
├── multisensorLmb/          # Multi-sensor LMB variants
│   ├── runParallelUpdateLmbFilter.m  # PU/GA/AA-LMB filters
│   ├── runIcLmbFilter.m              # Iterated-corrector LMB
│   ├── puLmbTrackMerging.m           # Parallel update fusion
│   ├── gaLmbTrackMerging.m           # Geometric average fusion
│   └── aaLmbTrackMerging.m           # Arithmetic average fusion
├── multisensorLmbm/         # Multi-sensor LMBM filter
│   └── runMultisensorLmbmFilter.m    # Exact multi-sensor LMBM
├── marginalEvaluations/     # Data association comparisons
├── trials/                  # Performance evaluation scripts
├── runFilters.m             # Quick start: single-sensor demo
├── runMultisensorFilters.m  # Quick start: multi-sensor demo
└── setPath.m                # Path setup script
```

### Data Flow

```
User → Model Generation (generateModel/generateMultisensorModel)
     → Ground Truth Simulation (generateGroundTruth)
     → Measurements (noisy observations + clutter)
     → Filter (runLmbFilter/runLmbmFilter/etc.)
        ├── Prediction Step (motion model)
        └── Update Step (measurement association + correction)
            ├── Data Association (LBP/Gibbs/Murty)
            └── Gaussian Mixture Update
     → State Estimates (object positions, velocities, labels)
     → Performance Metrics (OSPA)
     → Visualization (plotResults)
```

## Algorithm Basics

### Multi-Object Tracking Problem

The filters solve the problem of tracking multiple objects simultaneously given:
- **Noisy measurements** of object positions
- **Clutter** (false alarms from sensors)
- **Missed detections** (objects not always detected)
- **Unknown number of objects** (objects appear and disappear)

### Data Association Algorithms

Three methods solve the measurement-to-track association problem:

#### 1. Loopy Belief Propagation (LBP) - **Recommended**
- **Speed:** Fast (∼100-200ms per timestep)
- **Accuracy:** High - approximates marginal association probabilities
- **How it works:** Message passing on factor graph to compute approximate posteriors
- **Best for:** Real-time applications, large numbers of objects/measurements
- **Limitation:** Cannot generate full association events (LMBM incompatible)

#### 2. Gibbs Sampling
- **Speed:** Moderate (∼200-500ms per timestep)
- **Accuracy:** High - Monte Carlo approximation converges to true distribution
- **How it works:** MCMC sampling to generate likely data association hypotheses
- **Best for:** When exact association events needed (LMBM filters)
- **Limitation:** Slower than LBP, requires tuning number of samples

#### 3. Murty's Algorithm
- **Speed:** Moderate (∼150-400ms per timestep)
- **Accuracy:** Exact - finds K-best assignments deterministically
- **How it works:** Ranked optimal assignments via branch-and-bound
- **Best for:** When deterministic K-best associations desired
- **Limitation:** Exponential worst-case complexity with problem size

### Filter Types

#### Labelled Multi-Bernoulli (LMB)
- **Approximation:** Assumes independence between object existence and spatial distribution
- **Computational Cost:** Low - linear in number of objects and measurements
- **Accuracy:** Very good for most scenarios
- **Use Case:** Default choice for single-sensor tracking

#### LMB Mixture (LMBM)
- **Approximation:** None - exact closed-form solution to multi-object Bayes filter
- **Computational Cost:** Exponential in measurements (prohibitive for >10 objects)
- **Accuracy:** Theoretically optimal
- **Use Case:** Benchmarking, small-scale problems

### Multi-Sensor Fusion Strategies

When fusing information from multiple sensors:

#### Parallel Update (PU-LMB)
- **Accuracy:** Highest among approximate methods
- **Assumption:** Sensors are independent
- **Fusion:** Multiplicative combination of sensor likelihoods
- **Best for:** Independent sensors (e.g., different locations/modalities)

#### Geometric Average (GA-LMB)
- **Localization:** Excellent (accurate object positions)
- **Cardinality:** Poor (object count often wrong)
- **Assumption:** No independence assumption
- **Best for:** When position accuracy matters more than object count

#### Arithmetic Average (AA-LMB)
- **Localization:** Moderate
- **Cardinality:** Better than GA-LMB
- **Assumption:** No independence assumption
- **Best for:** Balanced position/cardinality performance

#### Iterated-Corrector (IC-LMB)
- **Method:** Sequential sensor processing (sensor 1 → 2 → 3...)
- **Performance:** Depends on sensor ordering
- **Best for:** Traditional multi-sensor implementations

## Inputs and Outputs

### Model Structure (Input)

Generated by `generateModel()` or `generateMultisensorModel()`, contains:

```matlab
model.xDimension            % State space dimension (4: [x, y, vx, vy])
model.zDimension            % Measurement dimension (2: [x, y])
model.A                     % State transition matrix (4×4)
model.R                     % Process noise covariance (4×4)
model.C                     % Observation matrix (2×4) or cell{sensor}
model.Q                     % Measurement noise covariance (2×2) or cell{sensor}
model.detectionProbability  % P(detection | object exists) [0-1]
model.survivalProbability   % P(survival) per timestep [0-1]
model.clutterRate           % Expected false alarms per timestep
model.dataAssociationMethod % 'LBP', 'Gibbs', or 'Murty'
model.numberOfBirthLocations % Number of potential object spawn points
model.rB                    % Birth probability at each location
model.muB                   % Birth mean states (cell array)
model.SigmaB                % Birth covariances (cell array)

% Multi-sensor specific:
model.numberOfSensors       % Number of sensors
model.lmbParallelUpdateMode % 'PU', 'GA', or 'AA'
```

### Measurements Structure (Input)

**Single-sensor:**
```matlab
measurements{t}  % 2×M matrix at time t (M measurements)
                 % Row 1: x-positions, Row 2: y-positions
                 % Mix of true detections + clutter
```

**Multi-sensor:**
```matlab
measurements{s, t}  % 2×M matrix from sensor s at time t
```

**Example:**
```matlab
measurements{1} = [12.3  15.1  -8.2;   % x-positions (3 measurements)
                    4.5   9.7  11.4]   % y-positions
```

### State Estimates Structure (Output)

Returned by `runLmbFilter()`, `runLmbmFilter()`, etc.:

```matlab
stateEstimates.labels{t}     % Cell array of object labels at time t
                             % Example: {[1, 1], [2, 1], [3, 1]}
                             % Format: [birth_time, birth_location]

stateEstimates.mu{t}         % Cell array of state vectors (4×1)
                             % Each: [x; y; vx; vy] (position + velocity)

stateEstimates.Sigma{t}      % Cell array of covariance matrices (4×4)
                             % Uncertainty ellipsoids for each object

stateEstimates.objects       % Full trajectory history struct with:
  .label                     % Object identifier
  .r                         % Existence probability
  .mu                        % State mean (cell array over time)
  .Sigma                     % State covariance (cell array over time)
  .w                         % Gaussian mixture weights (if applicable)
```

**Example output at time t=50:**
```matlab
stateEstimates.labels{50} = {[1,1], [5,2], [12,3]}  % 3 objects detected
stateEstimates.mu{50}{1} = [10.2; 5.3; 0.15; -0.22] % Object 1 state
stateEstimates.Sigma{50}{1} = 4×4 covariance matrix % Object 1 uncertainty
```

### Ground Truth Structure (Output)

Returned by `generateGroundTruth()`:

```matlab
groundTruth          % Cell array per object, each containing:
  {obj}.state{t}     % 4×1 true state at time t
  {obj}.startTime    % First timestep object exists
  {obj}.endTime      % Last timestep object exists

groundTruthRfs       % Random Finite Set representation:
  .mu{t}             % Cell array of true states at time t
  .Sigma{t}          % Cell array of covariances (for OSPA)
  .cardinality(t)    % Number of objects at time t
```

### Quick Reference: Common Operations

```matlab
% Create model
model = generateModel(clutterRate, detectionProb, 'LBP', 'Fixed');

% Generate scenario
[gt, meas, gtRfs] = generateGroundTruth(model);

% Run filter
est = runLmbFilter(model, meas);

% Access results at timestep t
numObjects = length(est.labels{t});
positions = cellfun(@(x) x(1:2), est.mu{t}, 'UniformOutput', false);

% Compute accuracy
[eOspa, hOspa, cardinality] = computeSimulationOspa(model, gtRfs, est);
```

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
