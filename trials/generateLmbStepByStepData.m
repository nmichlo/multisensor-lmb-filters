function generateLmbStepByStepData()
% GENERATELMBSTEPBYSTEPDATA -- Generate step-by-step algorithm data for LMB filter
%
% This script generates detailed intermediate state data for ALL steps of the
% LMB filter algorithm, enabling deep validation of Rust implementation.
%
% Captures for a single representative timestep:
%   1. Prediction step (inputs/outputs)
%   2. Association matrices generation (inputs/outputs)
%   3. Data association - LBP/Gibbs/Murty's (inputs/outputs)
%   4. Update step (inputs/outputs)
%   5. Cardinality estimation (inputs/outputs)
%
% Output: fixtures/step_by_step/lmb_step_by_step_seed42.json

%% Admin
clc; close all;
pkg load statistics; % For Octave
setPath;

%% Configuration
seed = 42;
timestepToCapture = 5;  % Capture data from timestep 5 (objects exist, varied detections)

%% Generate model
fprintf('Generating model...\n');
modelRng = SimpleRng(0);
[modelRng, model] = generateModel(modelRng, 10, 0.8, 'LBP');  % Higher clutter, imperfect detection

%% Generate ground truth (run to timestep 5)
fprintf('Generating ground truth...\n');
trialRng = SimpleRng(seed);
[trialRng, groundTruth, measurements, groundTruthRfs] = generateGroundTruth(trialRng, model);
fprintf('Ground truth generated (%d timesteps)\n', numel(measurements));

%% Run filter up to capture timestep to get realistic prior state
fprintf('Running filter to timestep %d...\n', timestepToCapture);
objects = model.object;  % Start with births
filterRng = SimpleRng(seed + 1000);

for t = 1:(timestepToCapture - 1)
    % Prediction
    objects = lmbPredictionStep(objects, model, t);
    % Update
    if (numel(measurements{t}))
        [associationMatrices, posteriorParameters] = generateLmbAssociationMatrices(objects, measurements{t}, model);
        if (strcmp(model.dataAssociationMethod, 'LBP'))
            [r, W] = loopyBeliefPropagation(associationMatrices, model.lbpConvergenceTolerance, model.maximumNumberOfLbpIterations);
        elseif(strcmp(model.dataAssociationMethod, 'Gibbs'))
            [filterRng, r, W] = lmbGibbsSampling(filterRng, associationMatrices, model.numberOfSamples);
        else
            [r, W] = lmbMurtysAlgorithm(associationMatrices, model.numberOfAssignments);
        end
        objects = computePosteriorLmbSpatialDistributions(objects, r, W, posteriorParameters, model);
    else
        for i = 1:numel(objects)
            objects(i).r = (objects(i).r * (1-model.detectionProbability)) / (1 - objects(i).r * model.detectionProbability);
        end
    end
    % Gate tracks
    objectsLikelyToExist = [objects.r] > model.existenceThreshold;
    objects = objects(objectsLikelyToExist);
end

fprintf('Filter state at timestep %d: %d objects\n', timestepToCapture-1, numel(objects));

%% Now capture step-by-step data for timestep t
t = timestepToCapture;
fprintf('\n=== Capturing step-by-step data for timestep %d ===\n', t);

fixtureData = struct();
fixtureData.seed = seed;
fixtureData.timestep = t;
fixtureData.model = captureModelData(model);
fixtureData.measurements = measurements{t};

%% Step 1: Prediction
fprintf('  [1/5] Capturing prediction step...\n');

% Input: prior objects (from t-1)
predictionInput = struct();
predictionInput.prior_objects = captureObjectsData(objects);
predictionInput.model_A = model.A;  % State transition matrix
predictionInput.model_R = model.R;  % Process noise covariance
predictionInput.model_P_s = model.survivalProbability;
predictionInput.timestep = t;

% Execute prediction
predictedObjects = lmbPredictionStep(objects, model, t);

% Output: predicted objects
predictionOutput = struct();
predictionOutput.predicted_objects = captureObjectsData(predictedObjects);

fixtureData.step1_prediction = struct();
fixtureData.step1_prediction.input = predictionInput;
fixtureData.step1_prediction.output = predictionOutput;

%% Step 2: Association matrices
fprintf('  [2/5] Capturing association matrices generation...\n');

% Input: predicted objects, measurements, model params
associationInput = struct();
associationInput.predicted_objects = captureObjectsData(predictedObjects);
associationInput.measurements = measurements{t};
associationInput.model_C = model.C;  % Observation matrix
associationInput.model_Q = model.Q;  % Measurement noise covariance
associationInput.model_P_d = model.detectionProbability;
associationInput.model_clutter = model.clutterPerUnitVolume;

% Execute association matrices generation
[associationMatrices, posteriorParameters] = generateLmbAssociationMatrices(predictedObjects, measurements{t}, model);

% Output: association matrices
associationOutput = struct();
associationOutput.C = associationMatrices.C;
associationOutput.L = associationMatrices.L;
associationOutput.R = associationMatrices.R;
associationOutput.P = associationMatrices.P;
associationOutput.eta = associationMatrices.eta;
associationOutput.posteriorParameters = capturePosteriorParams(posteriorParameters);

fixtureData.step2_association = struct();
fixtureData.step2_association.input = associationInput;
fixtureData.step2_association.output = associationOutput;

%% Step 3a: Data association - LBP
fprintf('  [3a/5] Capturing LBP data association...\n');

lbpInput = struct();
lbpInput.C = associationMatrices.C;
lbpInput.L = associationMatrices.L;
lbpInput.R = associationMatrices.R;
lbpInput.P = associationMatrices.P;
lbpInput.eta = associationMatrices.eta;
lbpInput.convergence_tolerance = model.lbpConvergenceTolerance;
lbpInput.max_iterations = model.maximumNumberOfLbpIterations;

[r_lbp, W_lbp] = loopyBeliefPropagation(associationMatrices, model.lbpConvergenceTolerance, model.maximumNumberOfLbpIterations);

lbpOutput = struct();
lbpOutput.r = r_lbp;
lbpOutput.W = W_lbp;

fixtureData.step3a_lbp = struct();
fixtureData.step3a_lbp.input = lbpInput;
fixtureData.step3a_lbp.output = lbpOutput;

%% Step 3b: Data association - Gibbs
fprintf('  [3b/5] Capturing Gibbs data association...\n');

gibbsInput = struct();
gibbsInput.C = associationMatrices.C;
gibbsInput.L = associationMatrices.L;
gibbsInput.R = associationMatrices.R;
gibbsInput.P = associationMatrices.P;
gibbsInput.eta = associationMatrices.eta;
gibbsInput.numberOfSamples = model.numberOfSamples;
gibbsInput.rng_seed = seed + 2000;  % Fixed seed for reproducibility

gibbsRng = SimpleRng(seed + 2000);
[gibbsRng, r_gibbs, W_gibbs] = lmbGibbsSampling(gibbsRng, associationMatrices, model.numberOfSamples);

gibbsOutput = struct();
gibbsOutput.r = r_gibbs;
gibbsOutput.W = W_gibbs;

fixtureData.step3b_gibbs = struct();
fixtureData.step3b_gibbs.input = gibbsInput;
fixtureData.step3b_gibbs.output = gibbsOutput;

%% Step 3c: Data association - Murty's
fprintf('  [3c/5] Capturing Murty''s data association...\n');

murtysInput = struct();
murtysInput.C = associationMatrices.C;
murtysInput.L = associationMatrices.L;
murtysInput.R = associationMatrices.R;
murtysInput.P = associationMatrices.P;
murtysInput.eta = associationMatrices.eta;
murtysInput.numberOfAssignments = model.numberOfAssignments;

[r_murtys, W_murtys] = lmbMurtysAlgorithm(associationMatrices, model.numberOfAssignments);

murtysOutput = struct();
murtysOutput.r = r_murtys;
murtysOutput.W = W_murtys;

fixtureData.step3c_murtys = struct();
fixtureData.step3c_murtys.input = murtysInput;
fixtureData.step3c_murtys.output = murtysOutput;

%% Step 4: Update (using LBP results)
fprintf('  [4/5] Capturing update step...\n');

updateInput = struct();
updateInput.predicted_objects = captureObjectsData(predictedObjects);
updateInput.r = r_lbp;
updateInput.W = W_lbp;
updateInput.posteriorParameters = capturePosteriorParams(posteriorParameters);
updateInput.model_C = model.C;  % Observation matrix
updateInput.model_Q = model.Q;  % Measurement noise covariance

posteriorObjects = computePosteriorLmbSpatialDistributions(predictedObjects, r_lbp, W_lbp, posteriorParameters, model);

updateOutput = struct();
updateOutput.posterior_objects = captureObjectsData(posteriorObjects);

fixtureData.step4_update = struct();
fixtureData.step4_update.input = updateInput;
fixtureData.step4_update.output = updateOutput;

%% Step 5: Cardinality estimation
fprintf('  [5/5] Capturing cardinality estimation...\n');

cardinalityInput = struct();
cardinalityInput.existence_probs = [posteriorObjects.r];

[nMap, mapIndices] = lmbMapCardinalityEstimate([posteriorObjects.r]);

cardinalityOutput = struct();
cardinalityOutput.n_estimated = nMap;
cardinalityOutput.map_indices = mapIndices;

fixtureData.step5_cardinality = struct();
fixtureData.step5_cardinality.input = cardinalityInput;
fixtureData.step5_cardinality.output = cardinalityOutput;

%% Create fixtures directory
fixturesDir = fullfile(fileparts(mfilename('fullpath')), '..', 'fixtures', 'step_by_step');
if ~exist(fixturesDir, 'dir')
    mkdir(fixturesDir);
    fprintf('Created directory: %s\n', fixturesDir);
end

%% Save to JSON file
filename = fullfile(fixturesDir, 'lmb_step_by_step_seed42.json');
fprintf('Saving to: %s\n', filename);

jsonStr = jsonencode(fixtureData);
fid = fopen(filename, 'w');
if fid == -1
    error('Failed to open file: %s', filename);
end
fprintf(fid, '%s', jsonStr);
fclose(fid);

fprintf('\n✓ LMB step-by-step fixture complete\n');
fprintf('Output: %s\n', filename);
fprintf('  - Timestep: %d\n', t);
fprintf('  - Objects: %d\n', numel(predictedObjects));
fprintf('  - Measurements: %d\n', size(measurements{t}, 2));

%%% Helper functions %%%

function modelData = captureModelData(model)
    % Capture essential model parameters
    modelData = struct();
    modelData.A = model.A;  % State transition matrix (F in Kalman notation)
    modelData.R = model.R;  % Process noise covariance (Q in Kalman notation)
    modelData.C = model.C;  % Observation matrix (H in Kalman notation)
    modelData.Q = model.Q;  % Measurement noise covariance (R in Kalman notation)
    modelData.P_s = model.survivalProbability;
    modelData.P_d = model.detectionProbability;
    modelData.clutter_per_unit_volume = model.clutterPerUnitVolume;
end

function objData = captureObjectsData(objects)
    % Capture object array data
    n = numel(objects);
    objData = cell(1, n);
    for i = 1:n
        obj = struct();
        obj.r = objects(i).r;
        obj.label = [objects(i).birthTime; objects(i).birthLocation];
        % Capture all GM components
        obj.mu = objects(i).mu;
        obj.Sigma = objects(i).Sigma;
        obj.w = objects(i).w;
        objData{i} = obj;
    end
end

function postData = capturePosteriorParams(posteriorParameters)
    % Capture posterior parameters
    n = numel(posteriorParameters);
    postData = cell(1, n);
    for i = 1:n
        post = struct();
        post.mu = posteriorParameters(i).mu;
        post.Sigma = posteriorParameters(i).Sigma;
        post.w = posteriorParameters(i).w;
        postData{i} = post;
    end
end

end  % End of main function
