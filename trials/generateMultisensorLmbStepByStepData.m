function generateMultisensorLmbStepByStepData()
% GENERATEMULTISENSORLMBSTEPBYSTEPDATA -- Generate step-by-step data for multisensor LMB filters
%
% This script generates detailed intermediate state data for multisensor LMB filter (IC-LMB),
% enabling deep validation of Rust implementation.
%
% Captures for a single representative timestep with 2 sensors:
%   1. Prediction step (inputs/outputs)
%   2. Sensor 1 update (association matrices, data association, update)
%   3. Sensor 2 update (association matrices, data association, update)
%   4. Cardinality estimation (inputs/outputs)
%
% Output: fixtures/step_by_step/multisensor_lmb_step_by_step_seed42.json

%% Admin
clc; close all;
pkg load statistics; % For Octave
setPath;

%% Configuration
seed = 42;
timestepToCapture = 3;  % Capture data from timestep 3
numberOfSensors = 2;  % Use 2 sensors for manageable complexity
clutterRates = [5 8];
detectionProbabilities = [0.7 0.8];
q = [3 4];

%% Generate multisensor model
fprintf('Generating multisensor model...\n');
modelRng = SimpleRng(0);
[modelRng, model] = generateMultisensorModel(modelRng, numberOfSensors, clutterRates, detectionProbabilities, q, 'IC', 'LBP', 'Fixed');

%% Generate ground truth
fprintf('Generating ground truth...\n');
trialRng = SimpleRng(seed);
[trialRng, groundTruth, measurements, groundTruthRfs] = generateMultisensorGroundTruth(trialRng, model);
fprintf('Ground truth generated (%d timesteps)\n', size(measurements, 2));

%% Run IC-LMB filter up to capture timestep
fprintf('Running IC-LMB filter to timestep %d...\n', timestepToCapture);
objects = model.object;

for t = 1:(timestepToCapture - 1)
    % Prediction
    objects = lmbPredictionStep(objects, model, t);
    % Per-sensor updates
    for s = 1:model.numberOfSensors
        if (numel(measurements{s, t}))
            [associationMatrices, posteriorParameters] = generateLmbSensorAssociationMatrices(objects, measurements{s, t}, model, s);
            [r, W] = loopyBeliefPropagation(associationMatrices, model.lbpConvergenceTolerance, model.maximumNumberOfLbpIterations);
            objects = computePosteriorLmbSpatialDistributions(objects, r, W, posteriorParameters, model);
        else
            for i = 1:numel(objects)
                objects(i).r = (objects(i).r * (1 - model.detectionProbability(s))) / (1 - objects(i).r * model.detectionProbability(s));
            end
        end
    end
    % Gate tracks
    objectsLikelyToExist = [objects.r] > model.existenceThreshold;
    objects = objects(objectsLikelyToExist);
end

fprintf('Filter state at timestep %d: %d objects\n', timestepToCapture-1, numel(objects));

%% Now capture step-by-step data for timestep t
t = timestepToCapture;
fprintf('\n=== Capturing IC-LMB step-by-step data for timestep %d ===\n', t);

fixtureData = struct();
fixtureData.seed = seed;
fixtureData.timestep = t;
fixtureData.filterType = 'IC-LMB';
fixtureData.numberOfSensors = numberOfSensors;
fixtureData.model = captureMultisensorModelData(model);
fixtureData.measurements = cell(1, numberOfSensors);
for s = 1:numberOfSensors
    fixtureData.measurements{s} = measurements{s, t};
end

%% Step 1: Prediction
fprintf('  [1/%d] Capturing prediction step...\n', 2 + 2*numberOfSensors);

predictionInput = struct();
predictionInput.prior_objects = captureObjectsData(objects);
predictionInput.model_A = model.A;
predictionInput.model_R = model.R;
predictionInput.model_P_s = model.survivalProbability;
predictionInput.timestep = t;

predictedObjects = lmbPredictionStep(objects, model, t);

predictionOutput = struct();
predictionOutput.predicted_objects = captureObjectsData(predictedObjects);

fixtureData.step1_prediction = struct();
fixtureData.step1_prediction.input = predictionInput;
fixtureData.step1_prediction.output = predictionOutput;

%% Per-sensor updates
fixtureData.sensorUpdates = cell(1, numberOfSensors);
currentObjects = predictedObjects;

for s = 1:numberOfSensors
    stepNum = 1 + s;
    fprintf('  [%d/%d] Capturing sensor %d update...\n', stepNum, 2 + 2*numberOfSensors, s);

    sensorUpdate = struct();
    sensorUpdate.sensorIndex = s;

    % Input: current objects (predicted for sensor 1, updated for sensor 2+)
    sensorUpdate.input = struct();
    sensorUpdate.input.objects = captureObjectsData(currentObjects);
    sensorUpdate.input.measurements = measurements{s, t};
    sensorUpdate.input.model_C = model.C;
    sensorUpdate.input.model_Q{s} = model.Q{s};
    sensorUpdate.input.model_P_d = model.detectionProbability(s);
    sensorUpdate.input.model_clutter = model.clutterPerUnitVolume(s);

    if (numel(measurements{s, t}))
        % Association matrices
        [associationMatrices, posteriorParameters] = generateLmbSensorAssociationMatrices(currentObjects, measurements{s, t}, model, s);

        sensorUpdate.association = struct();
        sensorUpdate.association.C = associationMatrices.C;
        sensorUpdate.association.L = associationMatrices.L;
        sensorUpdate.association.R = associationMatrices.R;
        sensorUpdate.association.P = associationMatrices.P;
        sensorUpdate.association.eta = associationMatrices.eta;
        sensorUpdate.association.posteriorParameters = capturePosteriorParams(posteriorParameters);

        % Data association (LBP)
        [r, W] = loopyBeliefPropagation(associationMatrices, model.lbpConvergenceTolerance, model.maximumNumberOfLbpIterations);

        sensorUpdate.dataAssociation = struct();
        sensorUpdate.dataAssociation.r = r;
        sensorUpdate.dataAssociation.W = W;

        % Update
        currentObjects = computePosteriorLmbSpatialDistributions(currentObjects, r, W, posteriorParameters, model);

        sensorUpdate.output = struct();
        sensorUpdate.output.updated_objects = captureObjectsData(currentObjects);
    else
        % No measurements - missed detection update
        for i = 1:numel(currentObjects)
            currentObjects(i).r = (currentObjects(i).r * (1 - model.detectionProbability(s))) / (1 - currentObjects(i).r * model.detectionProbability(s));
        end

        sensorUpdate.output = struct();
        sensorUpdate.output.updated_objects = captureObjectsData(currentObjects);
        sensorUpdate.association = [];
        sensorUpdate.dataAssociation = [];
    end

    fixtureData.sensorUpdates{s} = sensorUpdate;
end

%% Step final: Cardinality estimation
fprintf('  [%d/%d] Capturing cardinality estimation...\n', 2 + 2*numberOfSensors, 2 + 2*numberOfSensors);

cardinalityInput = struct();
cardinalityInput.existence_probs = [currentObjects.r];

[nMap, mapIndices] = lmbMapCardinalityEstimate([currentObjects.r]);

cardinalityOutput = struct();
cardinalityOutput.n_estimated = nMap;
cardinalityOutput.map_indices = mapIndices;

fixtureData.stepFinal_cardinality = struct();
fixtureData.stepFinal_cardinality.input = cardinalityInput;
fixtureData.stepFinal_cardinality.output = cardinalityOutput;

%% Create fixtures directory
fixturesDir = fullfile(fileparts(mfilename('fullpath')), '..', 'fixtures', 'step_by_step');
if ~exist(fixturesDir, 'dir')
    mkdir(fixturesDir);
    fprintf('Created directory: %s\n', fixturesDir);
end

%% Save to JSON file
filename = fullfile(fixturesDir, 'multisensor_lmb_step_by_step_seed42.json');
fprintf('Saving to: %s\n', filename);

jsonStr = jsonencode(fixtureData);
fid = fopen(filename, 'w');
if fid == -1
    error('Failed to open file: %s', filename);
end
fprintf(fid, '%s', jsonStr);
fclose(fid);

fprintf('\n✓ Multisensor LMB step-by-step fixture complete\n');
fprintf('Output: %s\n', filename);
fprintf('  - Filter type: IC-LMB\n');
fprintf('  - Timestep: %d\n', t);
fprintf('  - Sensors: %d\n', numberOfSensors);
fprintf('  - Objects (predicted): %d\n', numel(predictedObjects));
fprintf('  - Objects (final): %d\n', numel(currentObjects));

%%% Helper functions %%%

function modelData = captureMultisensorModelData(model)
    modelData = struct();
    modelData.A = model.A;
    modelData.R = model.R;
    modelData.C = model.C;
    modelData.Q = model.Q;  % Cell array, one per sensor
    modelData.P_s = model.survivalProbability;
    modelData.P_d = model.detectionProbability;  % Vector, one per sensor
    modelData.clutter_per_unit_volume = model.clutterPerUnitVolume;  % Vector, one per sensor
    modelData.numberOfSensors = model.numberOfSensors;
end

function objData = captureObjectsData(objects)
    n = numel(objects);
    objData = cell(1, n);
    for i = 1:n
        obj = struct();
        obj.r = objects(i).r;
        obj.label = [objects(i).birthTime; objects(i).birthLocation];
        obj.mu = objects(i).mu;
        obj.Sigma = objects(i).Sigma;
        obj.w = objects(i).w;
        objData{i} = obj;
    end
end

function postData = capturePosteriorParams(posteriorParameters)
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
