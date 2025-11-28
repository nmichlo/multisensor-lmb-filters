% GENERATEMULTISENSORLMBMDEBUGFIXTURE -- Debug fixture for Bug #19 investigation
%
% Generates detailed intermediate values from multi-sensor LMBM filter at t=0
% to identify divergence point between MATLAB and Rust implementations.
%
% Output: Console output with all intermediate values

%% Admin
clc; close all;
% pkg load statistics; % For Octave (not needed for this debug script)
setPath;

%% Configuration - match test setup exactly
seed = 1;  % Use seed 1 (known to produce cardinality 1 in MATLAB, 2 in Rust)
numberOfSensors = 3;
clutterRates = [5 5 5];
detectionProbabilities = [0.67 0.70 0.73];
q = [4 3 2];
lmbmSimulationLength = 10;

fprintf('=== MULTI-SENSOR LMBM DEBUG FIXTURE ===\n');
fprintf('Seed: %d\n', seed);
fprintf('Number of sensors: %d\n', numberOfSensors);

%% Generate model (same for all trials, fixed seed 0)
fprintf('\n=== MODEL GENERATION (seed 0) ===\n');
modelRng = SimpleRng(0);
[modelRng, model] = generateMultisensorModel(modelRng, numberOfSensors, clutterRates, detectionProbabilities, q, 'PU', 'LBP', 'Fixed');

fprintf('Number of birth locations: %d\n', model.numberOfBirthLocations);
fprintf('Birth r values (rBLmbm): [');
fprintf('%.6f ', model.rBLmbm);
fprintf(']\n');
fprintf('Detection probabilities: [');
fprintf('%.4f ', model.detectionProbability);
fprintf(']\n');
fprintf('Survival probability: %.4f\n', model.survivalProbability);
fprintf('Existence threshold: %.6f\n', model.existenceThreshold);
fprintf('Max posterior hypotheses: %d\n', model.maximumNumberOfPosteriorHypotheses);
fprintf('Hypothesis weight threshold: %.6e\n', model.posteriorHypothesisWeightThreshold);

%% Generate ground truth
fprintf('\n=== GROUND TRUTH GENERATION (seed %d) ===\n', seed);
trialRng = SimpleRng(seed);
[trialRng, groundTruth, measurements, groundTruthRfs] = generateMultisensorGroundTruth(trialRng, model);

% Truncate measurements for LMBM
measurementsShort = measurements(:, 1:lmbmSimulationLength);

fprintf('Measurements at t=1 (per sensor):\n');
for s = 1:numberOfSensors
    fprintf('  Sensor %d: %d measurements\n', s, size(measurementsShort{s, 1}, 2));
end

%% Run filter with detailed debug output
fprintf('\n=== FILTER EXECUTION (seed %d + 1000) ===\n', seed);
filterRng = SimpleRng(seed + 1000);

% Initialize (matching runMultisensorLmbmFilter.m)
simulationLength = lmbmSimulationLength;
hypotheses = model.hypotheses;
objects = model.trajectory;

fprintf('\n--- INITIAL STATE ---\n');
fprintf('Number of initial hypotheses: %d\n', numel(hypotheses));
fprintf('Initial hypothesis weight: %.6f\n', hypotheses(1).w);
fprintf('Initial hypothesis number of objects: %d\n', numel(hypotheses(1).r));

%% Process t=1 (which is t=0 in 0-indexed Rust)
t = 1;
fprintf('\n=== TIMESTEP t=%d ===\n', t);

% Add birth trajectories
[model.birthTrajectory.birthTime] = deal(t);
objects(end+1:end+model.numberOfBirthLocations) = model.birthTrajectory;
fprintf('Added %d birth trajectories\n', model.numberOfBirthLocations);

% Preallocate posterior hypotheses
posteriorHypotheses = repmat(model.hypotheses, 0, 1);

% Check measurements
measurementsAreAvailable = false;
for s = 1:model.numberOfSensors
    measurementsAreAvailable = measurementsAreAvailable || (numel(measurementsShort{s, t}) > 0);
end
fprintf('Measurements available: %d\n', measurementsAreAvailable);

% Process each prior hypothesis
fprintf('\n--- PRIOR HYPOTHESIS PROCESSING ---\n');
for i = 1:numel(hypotheses)
    fprintf('Processing prior hypothesis %d\n', i);

    % Prediction step
    priorHypothesis = lmbmPredictionStep(hypotheses(i), model, t);

    fprintf('After prediction:\n');
    fprintf('  Number of objects: %d\n', numel(priorHypothesis.r));
    fprintf('  r values: [');
    fprintf('%.6f ', priorHypothesis.r);
    fprintf(']\n');
    fprintf('  birth locations: [');
    fprintf('%d ', priorHypothesis.birthLocation);
    fprintf(']\n');
    fprintf('  birth times: [');
    fprintf('%d ', priorHypothesis.birthTime);
    fprintf(']\n');

    % Measurement update
    if measurementsAreAvailable
        % Generate association matrices
        [L, posteriorParameters] = generateMultisensorLmbmAssociationMatrices(priorHypothesis, measurementsShort(:, t), model);

        fprintf('Association matrices (L):\n');
        fprintf('  Dimensions: %s\n', mat2str(size(L)));
        fprintf('  Non-zero elements: %d\n', nnz(L));

        % Gibbs sampling
        [filterRng, A] = multisensorLmbmGibbsSampling(filterRng, L, model.numberOfSamples);

        fprintf('Gibbs sampling output (A):\n');
        fprintf('  Number of samples: %d\n', size(A, 1));
        fprintf('  First 5 samples:\n');
        for sampleIdx = 1:min(5, size(A, 1))
            fprintf('    Sample %d: [', sampleIdx);
            fprintf('%d ', A(sampleIdx, :));
            fprintf(']\n');
        end

        % Determine posterior hypothesis parameters
        newHypotheses = determineMultisensorPosteriorHypothesisParameters(A, L, posteriorParameters, priorHypothesis);

        fprintf('New hypotheses:\n');
        fprintf('  Number: %d\n', numel(newHypotheses));
        for h = 1:min(5, numel(newHypotheses))
            fprintf('  Hypothesis %d: w=%.6e, r=[', h, newHypotheses(h).w);
            fprintf('%.4f ', newHypotheses(h).r);
            fprintf(']\n');
        end

        posteriorHypotheses(end+1:end+numel(newHypotheses)) = newHypotheses;
    else
        priorHypothesis.r = (prod(1-model.detectionProbability) * priorHypothesis.r) ./ (1 - priorHypothesis.r + prod(1-model.detectionProbability) * priorHypothesis.r);
        posteriorHypotheses(end+1) = priorHypothesis;
    end
end

fprintf('\n--- BEFORE NORMALIZATION ---\n');
fprintf('Total posterior hypotheses: %d\n', numel(posteriorHypotheses));

% Normalization and gating
fprintf('\n--- NORMALIZATION AND GATING ---\n');
[hypotheses, objectsLikelyToExist] = lmbmNormalisationAndGating(posteriorHypotheses, model);

fprintf('After normalization:\n');
fprintf('  Number of hypotheses: %d\n', numel(hypotheses));
fprintf('  Objects likely to exist: [');
fprintf('%d ', objectsLikelyToExist);
fprintf(']\n');

for h = 1:numel(hypotheses)
    fprintf('  Hypothesis %d: w=%.6e, r=[', h, hypotheses(h).w);
    fprintf('%.4f ', hypotheses(h).r);
    fprintf(']\n');
end

% Compute r_total for debugging
fprintf('\n--- r_TOTAL COMPUTATION ---\n');
if numel(hypotheses) > 0 && numel(hypotheses(1).r) > 0
    numberOfObjects = numel(hypotheses(1).r);
    rTotal = zeros(numberOfObjects, 1);
    for h = 1:numel(hypotheses)
        rTotal = rTotal + hypotheses(h).w * hypotheses(h).r;
    end
    fprintf('r_total values: [');
    fprintf('%.6f ', rTotal);
    fprintf(']\n');
end

% State extraction
fprintf('\n--- STATE EXTRACTION ---\n');
[cardinalityEstimate, extractionIndices] = lmbmStateExtraction(hypotheses, false);

fprintf('MAP cardinality estimate: %d\n', cardinalityEstimate);
fprintf('Extraction indices: [');
fprintf('%d ', extractionIndices);
fprintf(']\n');

% Show extracted states
fprintf('\nExtracted states:\n');
for i = 1:cardinalityEstimate
    j = extractionIndices(i);
    fprintf('  Object %d (index %d): mu = [%.4f, %.4f, %.4f, %.4f]\n', i, j, ...
        hypotheses(1).mu{j}(1), hypotheses(1).mu{j}(2), hypotheses(1).mu{j}(3), hypotheses(1).mu{j}(4));
end

fprintf('\n=== END DEBUG OUTPUT ===\n');
