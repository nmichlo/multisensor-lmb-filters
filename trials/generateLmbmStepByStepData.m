function generateLmbmStepByStepData()
% GENERATELMBMSTEPBYSTEPDATA -- Generate step-by-step algorithm data for LMBM filter
%
% This script generates detailed intermediate state data for ALL steps of the
% LMBM filter algorithm, enabling deep validation of Rust implementation.
%
% Captures for a single representative timestep and hypothesis:
%   1. Prediction step (inputs/outputs)
%   2. Association matrices generation (inputs/outputs)
%   3. Gibbs sampling (inputs/outputs)
%   4. Hypothesis parameters (inputs/outputs)
%   5. Normalization and gating (inputs/outputs)
%   6. State extraction (inputs/outputs)
%
% Output: fixtures/step_by_step/lmbm_step_by_step_seed42.json

%% Admin
clc; close all;
pkg load statistics; % For Octave
setPath;

%% Configuration
seed = 42;
timestepToCapture = 3;  % Capture data from timestep 3 (hypotheses exist, manageable size)

%% Generate model
fprintf('Generating model...\n');
modelRng = SimpleRng(0);
[modelRng, model] = generateModel(modelRng, 10, 0.8, 'Gibbs');  % Higher clutter, Gibbs sampling

%% Generate ground truth
fprintf('Generating ground truth...\n');
trialRng = SimpleRng(seed);
[trialRng, groundTruth, measurements, groundTruthRfs] = generateGroundTruth(trialRng, model);
fprintf('Ground truth generated (%d timesteps)\n', numel(measurements));

%% Run filter up to capture timestep to get realistic prior state
fprintf('Running filter to timestep %d...\n', timestepToCapture);
hypotheses = model.hypotheses;
objects = model.trajectory;
filterRng = SimpleRng(seed + 1000);

for t = 1:(timestepToCapture - 1)
    % Add new trajectories
    [model.birthTrajectory.birthTime] = deal(t);
    objects(end+1:end+model.numberOfBirthLocations) = model.birthTrajectory;

    % Preallocate posterior hypotheses
    posteriorHypotheses = repmat(model.hypotheses, 0, 1);

    % Generate posterior hypotheses for each prior hypothesis
    for i = 1:numel(hypotheses)
        % Prediction
        priorHypothesis = lmbmPredictionStep(hypotheses(i), model, t);
        % Update
        if (numel(measurements{t}))
            [associationMatrices, posteriorParameters] = generateLmbmAssociationMatrices(priorHypothesis, measurements{t}, model);
            if(strcmp(model.dataAssociationMethod, 'Murty'))
                V = murtysAlgorithmWrapper(associationMatrices.C, model.numberOfAssignments);
            else
                [filterRng, V] = lmbmGibbsSampling(filterRng, associationMatrices.P, associationMatrices.C, model.numberOfSamples);
            end
            newHypotheses = determinePosteriorHypothesisParameters(V, associationMatrices.L, posteriorParameters, priorHypothesis);
            posteriorHypotheses(end+1:end+numel(newHypotheses)) = newHypotheses;
        else
            priorHypothesis.r = ((1-model.detectionProbability) * priorHypothesis.r) ./ (1 - model.detectionProbability * priorHypothesis.r);
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
fprintf('\n=== Capturing step-by-step data for timestep %d ===\n', t);

fixtureData = struct();
fixtureData.seed = seed;
fixtureData.timestep = t;
fixtureData.model = captureModelData(model);
fixtureData.measurements = measurements{t};
fixtureData.priorHypothesisIndex = 1;  % Using first hypothesis for step-by-step capture

% Add new trajectories (this happens before prediction in LMBM)
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

%% Step 2: Association matrices
fprintf('  [2/6] Capturing association matrices generation...\n');

associationInput = struct();
associationInput.predicted_hypothesis = captureHypothesisData(predictedHypothesis);
associationInput.measurements = measurements{t};
associationInput.model_C = model.C;
associationInput.model_Q = model.Q;
associationInput.model_P_d = model.detectionProbability;
associationInput.model_clutter = model.clutterPerUnitVolume;

[associationMatrices, posteriorParameters] = generateLmbmAssociationMatrices(predictedHypothesis, measurements{t}, model);

associationOutput = struct();
associationOutput.C = associationMatrices.C;
associationOutput.L = associationMatrices.L;
associationOutput.P = associationMatrices.P;
associationOutput.posteriorParameters = capturePosteriorParams(posteriorParameters);

fixtureData.step2_association = struct();
fixtureData.step2_association.input = associationInput;
fixtureData.step2_association.output = associationOutput;

%% Step 3a: Gibbs sampling
fprintf('  [3a/6] Capturing Gibbs sampling...\n');

gibbsInput = struct();
gibbsInput.P = associationMatrices.P;
gibbsInput.C = associationMatrices.C;
gibbsInput.numberOfSamples = model.numberOfSamples;
gibbsInput.rng_seed = seed + 2000;

gibbsRng = SimpleRng(seed + 2000);
[gibbsRng, V_gibbs] = lmbmGibbsSampling(gibbsRng, associationMatrices.P, associationMatrices.C, model.numberOfSamples);

gibbsOutput = struct();
gibbsOutput.V = V_gibbs;

fixtureData.step3a_gibbs = struct();
fixtureData.step3a_gibbs.input = gibbsInput;
fixtureData.step3a_gibbs.output = gibbsOutput;

%% Step 3b: Murty's algorithm
fprintf('  [3b/6] Capturing Murty''s algorithm...\n');

murtysInput = struct();
murtysInput.C = associationMatrices.C;
murtysInput.numberOfAssignments = model.numberOfAssignments;

V_murtys = murtysAlgorithmWrapper(associationMatrices.C, model.numberOfAssignments);

murtysOutput = struct();
murtysOutput.V = V_murtys;

fixtureData.step3b_murtys = struct();
fixtureData.step3b_murtys.input = murtysInput;
fixtureData.step3b_murtys.output = murtysOutput;

%% Step 4: Hypothesis parameters (using Gibbs samples)
fprintf('  [4/6] Capturing hypothesis parameters determination...\n');

hypothesisInput = struct();
hypothesisInput.V = V_gibbs;
hypothesisInput.L = associationMatrices.L;
hypothesisInput.posteriorParameters = capturePosteriorParams(posteriorParameters);
hypothesisInput.predicted_hypothesis = captureHypothesisData(predictedHypothesis);

newHypotheses = determinePosteriorHypothesisParameters(V_gibbs, associationMatrices.L, posteriorParameters, predictedHypothesis);

hypothesisOutput = struct();
hypothesisOutput.new_hypotheses = captureHypothesesData(newHypotheses);

fixtureData.step4_hypothesis = struct();
fixtureData.step4_hypothesis.input = hypothesisInput;
fixtureData.step4_hypothesis.output = hypothesisOutput;

%% Step 5: Normalization and gating
fprintf('  [5/6] Capturing normalization and gating...\n');

% For this step, we need to combine hypotheses from all prior hypotheses
% For simplicity, we'll just use the hypotheses generated from the first prior
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
filename = fullfile(fixturesDir, 'lmbm_step_by_step_seed42.json');
fprintf('Saving to: %s\n', filename);

jsonStr = jsonencode(fixtureData);
fid = fopen(filename, 'w');
if fid == -1
    error('Failed to open file: %s', filename);
end
fprintf(fid, '%s', jsonStr);
fclose(fid);

fprintf('\n✓ LMBM step-by-step fixture complete\n');
fprintf('Output: %s\n', filename);
fprintf('  - Timestep: %d\n', t);
fprintf('  - Prior hypotheses: %d\n', numel(hypotheses));
fprintf('  - Posterior hypotheses: %d\n', numel(normalizedHypotheses));
fprintf('  - Measurements: %d\n', size(measurements{t}, 2));

%%% Helper functions %%%

function modelData = captureModelData(model)
    modelData = struct();
    modelData.A = model.A;
    modelData.R = model.R;
    modelData.C = model.C;
    modelData.Q = model.Q;
    modelData.P_s = model.survivalProbability;
    modelData.P_d = model.detectionProbability;
    modelData.clutter_per_unit_volume = model.clutterPerUnitVolume;
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

function postData = capturePosteriorParams(posteriorParameters)
    % For LMBM, posteriorParameters is a single struct with:
    % - r: existence probabilities (vector)
    % - mu: cell array (n_objects × n_measurements+1)
    % - Sigma: cell array of covariances
    postData = struct();
    postData.r = posteriorParameters.r;
    postData.mu = posteriorParameters.mu;
    postData.Sigma = posteriorParameters.Sigma;
end

end  % End of main function
