function generateMultisensorLmbmStepByStepData()
% GENERATEMULTISENSORLMBMSTEPBYSTEPDATA -- Generate step-by-step data for multisensor LMBM filter
%
% This script generates detailed intermediate state data for the multisensor LMBM filter,
% enabling deep validation of Rust implementation.
%
% Captures for a single representative timestep and hypothesis with 2 sensors:
%   1. Prediction step (inputs/outputs)
%   2. Multisensor association matrices generation (inputs/outputs)
%   3. Multisensor Gibbs sampling (inputs/outputs)
%   4. Multisensor hypothesis parameters (inputs/outputs)
%   5. Normalization and gating (inputs/outputs)
%   6. State extraction (inputs/outputs)
%
% Output: fixtures/step_by_step/multisensor_lmbm_step_by_step_seed42.json

%% Admin
clc; close all;
pkg load statistics; % For Octave
setPath;

%% Configuration
seed = 42;
timestepToCapture = 1;  % Capture data from timestep 1 (no warmup needed)
numberOfSensors = 2;  % Use 2 sensors for manageable complexity
clutterRates = [5 8];
detectionProbabilities = [0.7 0.8];
q = [3 4];

%% Generate multisensor model
fprintf('Generating multisensor model...\n');
modelRng = SimpleRng(0);
[modelRng, model] = generateMultisensorModel(modelRng, numberOfSensors, clutterRates, detectionProbabilities, q, 'IC', 'Gibbs', 'Fixed');

%% Generate ground truth
fprintf('Generating ground truth...\n');
trialRng = SimpleRng(seed);
[trialRng, groundTruth, measurements, groundTruthRfs] = generateMultisensorGroundTruth(trialRng, model);
fprintf('Ground truth generated (%d timesteps)\n', size(measurements, 2));

%% Run multisensor LMBM filter up to capture timestep
fprintf('Running multisensor LMBM filter to timestep %d...\n', timestepToCapture);
hypotheses = model.hypotheses;
objects = model.trajectory;
filterRng = SimpleRng(seed + 1000);  % Separate RNG for filter

for t = 1:(timestepToCapture - 1)
    % Add new trajectories
    [model.birthTrajectory.birthTime] = deal(t);
    objects(end+1:end+model.numberOfBirthLocations) = model.birthTrajectory;

    % Preallocate posterior hypotheses
    posteriorHypotheses = repmat(model.hypotheses, 0, 1);

    % Check for measurements
    measurementsAreAvailable = false;
    for s = 1:model.numberOfSensors
        measurementsAreAvailable = measurementsAreAvailable || (numel(measurements{s, t}) > 0);
    end

    % Generate posterior hypotheses for each prior hypothesis
    for i = 1:numel(hypotheses)
        % Prediction
        priorHypothesis = lmbmPredictionStep(hypotheses(i), model, t);
        % Update
        if (measurementsAreAvailable)
            [L, posteriorParameters] = generateMultisensorLmbmAssociationMatrices(priorHypothesis, measurements(:, t), model);
            [filterRng, A] = multisensorLmbmGibbsSampling(filterRng, L, model.numberOfSamples);
            newHypotheses = determineMultisensorPosteriorHypothesisParameters(A, L, posteriorParameters, priorHypothesis);
            posteriorHypotheses(end+1:end+numel(newHypotheses)) = newHypotheses;
        else
            priorHypothesis.r = (prod(1-model.detectionProbability) * priorHypothesis.r) ./ (1 - priorHypothesis.r + prod(1-model.detectionProbability) * priorHypothesis.r);
            posteriorHypotheses(end+1) = priorHypothesis;
        end
    end

    % Normalize and gate
    [hypotheses, objectsLikelyToExist] = lmbmNormalisationAndGating(posteriorHypotheses, model);
    objects = objects(objectsLikelyToExist);
end

fprintf('Filter state at timestep %d: %d hypotheses, %d objects\n', timestepToCapture-1, numel(hypotheses), numel(objects));

%% Now capture step-by-step data for timestep t (using first prior hypothesis)
t = timestepToCapture;
fprintf('\n=== Capturing multisensor LMBM step-by-step data for timestep %d ===\n', t);

fixtureData = struct();
fixtureData.seed = seed;
fixtureData.timestep = t;
fixtureData.filterType = 'Multisensor-LMBM';
fixtureData.numberOfSensors = numberOfSensors;
fixtureData.model = captureMultisensorModelData(model);
fixtureData.measurements = cell(1, numberOfSensors);
for s = 1:numberOfSensors
    fixtureData.measurements{s} = measurements{s, t};
end
fixtureData.priorHypothesisIndex = 1;  % Using first hypothesis

% Add new trajectories (happens before prediction in LMBM)
[model.birthTrajectory.birthTime] = deal(t);
objects(end+1:end+model.numberOfBirthLocations) = model.birthTrajectory;

%% Step 1: Prediction (for first hypothesis)
fprintf('  [1/6] Capturing prediction step (hypothesis 1)...\n');

predictionInput = struct();
predictionInput.prior_hypothesis = captureHypothesisData(hypotheses(1));
predictionInput.model_A = model.A;
predictionInput.model_R = model.R;
predictionInput.model_P_s = model.survivalProbability;
predictionInput.timestep = t;

predictedHypothesis = lmbmPredictionStep(hypotheses(1), model, t);

predictionOutput = struct();
predictionOutput.predicted_hypothesis = captureHypothesisData(predictedHypothesis);

fixtureData.step1_prediction = struct();
fixtureData.step1_prediction.input = predictionInput;
fixtureData.step1_prediction.output = predictionOutput;

%% Step 2: Multisensor association matrices
fprintf('  [2/6] Capturing multisensor association matrices generation...\n');

associationInput = struct();
associationInput.predicted_hypothesis = captureHypothesisData(predictedHypothesis);
associationInput.measurements = cell(1, numberOfSensors);
for s = 1:numberOfSensors
    associationInput.measurements{s} = measurements{s, t};
end
associationInput.model_C = model.C;
associationInput.model_Q = model.Q;  % Cell array
associationInput.model_P_d = model.detectionProbability;  % Vector
associationInput.model_clutter = model.clutterPerUnitVolume;  % Vector

[L, posteriorParameters] = generateMultisensorLmbmAssociationMatrices(predictedHypothesis, measurements(:, t), model);

associationOutput = struct();
associationOutput.L = L;
associationOutput.posteriorParameters = captureMultisensorPosteriorParams(posteriorParameters);

fixtureData.step2_association = struct();
fixtureData.step2_association.input = associationInput;
fixtureData.step2_association.output = associationOutput;

%% Step 3: Multisensor Gibbs sampling
fprintf('  [3/6] Capturing multisensor Gibbs sampling...\n');

gibbsInput = struct();
gibbsInput.L = L;
gibbsInput.numberOfSamples = model.numberOfSamples;
gibbsInput.rng_seed = seed + 2000;

gibbsRng = SimpleRng(seed + 2000);
[gibbsRng, A] = multisensorLmbmGibbsSampling(gibbsRng, L, model.numberOfSamples);

gibbsOutput = struct();
gibbsOutput.A = A;

fixtureData.step3_gibbs = struct();
fixtureData.step3_gibbs.input = gibbsInput;
fixtureData.step3_gibbs.output = gibbsOutput;

%% Step 4: Multisensor hypothesis parameters
fprintf('  [4/6] Capturing multisensor hypothesis parameters determination...\n');

hypothesisInput = struct();
hypothesisInput.A = A;
hypothesisInput.L = L;
hypothesisInput.posteriorParameters = captureMultisensorPosteriorParams(posteriorParameters);
hypothesisInput.predicted_hypothesis = captureHypothesisData(predictedHypothesis);

newHypotheses = determineMultisensorPosteriorHypothesisParameters(A, L, posteriorParameters, predictedHypothesis);

hypothesisOutput = struct();
hypothesisOutput.new_hypotheses = captureHypothesesData(newHypotheses);

fixtureData.step4_hypothesis = struct();
fixtureData.step4_hypothesis.input = hypothesisInput;
fixtureData.step4_hypothesis.output = hypothesisOutput;

%% Step 5: Normalization and gating
fprintf('  [5/6] Capturing normalization and gating...\n');

posteriorHypotheses = newHypotheses;

normalizationInput = struct();
normalizationInput.posterior_hypotheses = captureHypothesesData(posteriorHypotheses);
normalizationInput.model_posterior_hypothesis_weight_threshold = model.posteriorHypothesisWeightThreshold;
normalizationInput.model_maximum_number_of_posterior_hypotheses = model.maximumNumberOfPosteriorHypotheses;
normalizationInput.model_existence_threshold = model.existenceThreshold;

[normalizedHypotheses, objectsLikelyToExist] = lmbmNormalisationAndGating(posteriorHypotheses, model);

normalizationOutput = struct();
normalizationOutput.normalized_hypotheses = captureHypothesesData(normalizedHypotheses);
normalizationOutput.objects_likely_to_exist = objectsLikelyToExist;

fixtureData.step5_normalization = struct();
fixtureData.step5_normalization.input = normalizationInput;
fixtureData.step5_normalization.output = normalizationOutput;

%% Step 6: State extraction
fprintf('  [6/6] Capturing state extraction...\n');

extractionInput = struct();
extractionInput.hypotheses = captureHypothesesData(normalizedHypotheses);
extractionInput.use_map = false;  % EAP extraction

[cardinalityEstimate, extractionIndices] = lmbmStateExtraction(normalizedHypotheses, false);

extractionOutput = struct();
extractionOutput.cardinality_estimate = cardinalityEstimate;
extractionOutput.extraction_indices = extractionIndices;

fixtureData.step6_extraction = struct();
fixtureData.step6_extraction.input = extractionInput;
fixtureData.step6_extraction.output = extractionOutput;

%% Create fixtures directory
fixturesDir = fullfile(fileparts(mfilename('fullpath')), '..', 'fixtures', 'step_by_step');
if ~exist(fixturesDir, 'dir')
    mkdir(fixturesDir);
    fprintf('Created directory: %s\n', fixturesDir);
end

%% Save to JSON file
filename = fullfile(fixturesDir, 'multisensor_lmbm_step_by_step_seed42.json');
fprintf('Saving to: %s\n', filename);

jsonStr = jsonencode(fixtureData);
fid = fopen(filename, 'w');
if fid == -1
    error('Failed to open file: %s', filename);
end
fprintf(fid, '%s', jsonStr);
fclose(fid);

fprintf('\n✓ Multisensor LMBM step-by-step fixture complete\n');
fprintf('Output: %s\n', filename);
fprintf('  - Filter type: Multisensor-LMBM\n');
fprintf('  - Timestep: %d\n', t);
fprintf('  - Sensors: %d\n', numberOfSensors);
fprintf('  - Prior hypotheses: %d\n', numel(hypotheses));
fprintf('  - Posterior hypotheses: %d\n', numel(normalizedHypotheses));

%%% Helper functions %%%

function modelData = captureMultisensorModelData(model)
    modelData = struct();
    modelData.A = model.A;
    modelData.R = model.R;
    modelData.C = model.C;
    modelData.Q = model.Q;  % Cell array
    modelData.P_s = model.survivalProbability;
    modelData.P_d = model.detectionProbability;  % Vector
    modelData.clutter_per_unit_volume = model.clutterPerUnitVolume;  % Vector
    modelData.numberOfSensors = model.numberOfSensors;
end

function hypData = captureHypothesisData(hypothesis)
    hypData = struct();
    hypData.w = hypothesis.w;
    hypData.r = hypothesis.r;
    hypData.mu = hypothesis.mu;
    hypData.Sigma = hypothesis.Sigma;
    hypData.birthTime = hypothesis.birthTime;
    hypData.birthLocation = hypothesis.birthLocation;
end

function hypsData = captureHypothesesData(hypotheses)
    n = numel(hypotheses);
    hypsData = cell(1, n);
    for i = 1:n
        hypsData{i} = captureHypothesisData(hypotheses(i));
    end
end

function postData = captureMultisensorPosteriorParams(posteriorParameters)
    % For multisensor LMBM, posteriorParameters structure varies
    % This is a cell array of per-sensor parameters
    postData = posteriorParameters;  % Just pass through as-is
end

end  % End of main function
