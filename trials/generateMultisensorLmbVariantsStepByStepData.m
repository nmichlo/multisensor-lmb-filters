function generateMultisensorLmbVariantsStepByStepData()
% GENERATEMULTISENSORLMBVARIANTSSTEPBYSTEPDATA -- Generate step-by-step fixtures for ALL multi-sensor LMB variants
%
% This script generates detailed intermediate state data for all multi-sensor LMB filter variants:
%   - AA-LMB (Arithmetic Average)
%   - GA-LMB (Geometric Average)
%   - PU-LMB (Parallel Update)
%   - IC-LMB (Iterated Corrector)
%
% Each fixture captures for a single representative timestep with 2 sensors:
%   1. Prediction step (inputs/outputs)
%   2. Per-sensor association matrices (C, L, R, P, eta, posteriorParameters)
%   3. Per-sensor data association (r, W from LBP)
%   4. Per-sensor update (updated_objects)
%   5. Fusion step (fused_objects) - for parallel variants (AA/GA/PU)
%   6. Cardinality estimation (n_estimated, map_indices)
%
% Output: tests/data/step_by_step/{aa,ga,pu,ic}_lmb_step_by_step_seed42.json
%
% Usage (from multisensor-lmb-filters directory):
%   octave --eval "run('trials/generateMultisensorLmbVariantsStepByStepData.m')"

%% Admin
clc; close all;
pkg load statistics; % For Octave
setPath;

%% Configuration
seed = 42;
timestepToCapture = 3;  % Capture data from timestep 3
numberOfSensors = 2;    % Use 2 sensors for manageable complexity
clutterRates = [5 8];
detectionProbabilities = [0.7 0.8];
q = [3 4];  % Process noise std per sensor (not used directly, but affects model)

% Filter variants to generate
filterVariants = {'AA', 'GA', 'PU', 'IC'};

%% Generate fixtures for each variant
for vIdx = 1:numel(filterVariants)
    variant = filterVariants{vIdx};
    fprintf('\n========================================\n');
    fprintf('Generating %s-LMB step-by-step fixture\n', variant);
    fprintf('========================================\n');

    generateVariantFixture(variant, seed, timestepToCapture, numberOfSensors, clutterRates, detectionProbabilities, q);
end

fprintf('\n========================================\n');
fprintf('All fixtures generated successfully!\n');
fprintf('========================================\n');

end  % End of main function

%% =========================================================================
%% Main fixture generation for a single variant
%% =========================================================================

function generateVariantFixture(variant, seed, timestepToCapture, numberOfSensors, clutterRates, detectionProbabilities, q)

%% Generate multisensor model
fprintf('  Generating multisensor model...\n');
modelRng = SimpleRng(0);

% IC-LMB uses different internal processing but same model structure
if strcmp(variant, 'IC')
    [modelRng, model] = generateMultisensorModel(modelRng, numberOfSensors, clutterRates, detectionProbabilities, q, 'IC', 'LBP', 'Fixed');
else
    [modelRng, model] = generateMultisensorModel(modelRng, numberOfSensors, clutterRates, detectionProbabilities, q, variant, 'LBP', 'Fixed');
end

%% Generate ground truth
fprintf('  Generating ground truth...\n');
trialRng = SimpleRng(seed);
[trialRng, groundTruth, measurements, groundTruthRfs] = generateMultisensorGroundTruth(trialRng, model);
fprintf('  Ground truth generated (%d timesteps)\n', size(measurements, 2));

%% Run filter up to capture timestep to build up track state
fprintf('  Running %s-LMB filter to timestep %d...\n', variant, timestepToCapture);
objects = model.object;

for t = 1:(timestepToCapture - 1)
    % Prediction
    objects = lmbPredictionStep(objects, model, t);

    if strcmp(variant, 'IC')
        % IC-LMB: Sequential processing
        for s = 1:model.numberOfSensors
            if numel(measurements{s, t})
                [associationMatrices, posteriorParameters] = generateLmbSensorAssociationMatrices(objects, measurements{s, t}, model, s);
                [r, W] = loopyBeliefPropagation(associationMatrices, model.lbpConvergenceTolerance, model.maximumNumberOfLbpIterations);
                objects = computePosteriorLmbSpatialDistributions(objects, r, W, posteriorParameters, model);
            else
                for i = 1:numel(objects)
                    objects(i).r = (objects(i).r * (1 - model.detectionProbability(s))) / (1 - objects(i).r * model.detectionProbability(s));
                end
            end
        end
    else
        % AA/GA/PU: Parallel processing
        measurementUpdatedDistributions = cell(1, model.numberOfSensors);
        for s = 1:model.numberOfSensors
            if numel(measurements{s, t})
                [associationMatrices, posteriorParameters] = generateLmbSensorAssociationMatrices(objects, measurements{s, t}, model, s);
                [r, W] = loopyBeliefPropagation(associationMatrices, model.lbpConvergenceTolerance, model.maximumNumberOfLbpIterations);
                measurementUpdatedDistributions{s} = computePosteriorLmbSpatialDistributions(objects, r, W, posteriorParameters, model);
            else
                measurementUpdatedDistributions{s} = objects;
                for i = 1:numel(objects)
                    measurementUpdatedDistributions{s}(i).r = (measurementUpdatedDistributions{s}(i).r * (1 - model.detectionProbability(s))) / (1 - measurementUpdatedDistributions{s}(i).r * model.detectionProbability(s));
                end
            end
        end

        % Fusion
        if strcmp(variant, 'AA')
            objects = aaLmbTrackMerging(measurementUpdatedDistributions, model);
        elseif strcmp(variant, 'GA')
            objects = gaLmbTrackMerging(measurementUpdatedDistributions, model);
        else  % PU
            objects = puLmbTrackMerging(measurementUpdatedDistributions, objects, model);
        end
    end

    % Gate tracks
    objectsLikelyToExist = [objects.r] > model.existenceThreshold;
    objects = objects(objectsLikelyToExist);
end

fprintf('  Filter state at timestep %d: %d objects\n', timestepToCapture-1, numel(objects));

%% Now capture step-by-step data for timestep t
t = timestepToCapture;
fprintf('\n  === Capturing %s-LMB step-by-step data for timestep %d ===\n', variant, t);

fixtureData = struct();
fixtureData.seed = seed;
fixtureData.timestep = t;
fixtureData.filterType = [variant '-LMB'];
fixtureData.numberOfSensors = numberOfSensors;
fixtureData.model = captureMultisensorModelData(model);
fixtureData.measurements = cell(1, numberOfSensors);
for s = 1:numberOfSensors
    fixtureData.measurements{s} = measurements{s, t};
end

%% Step 1: Prediction
fprintf('  [1] Capturing prediction step...\n');

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

if strcmp(variant, 'IC')
    % IC-LMB: Sequential processing
    currentObjects = predictedObjects;

    for s = 1:numberOfSensors
        fprintf('  [%d] Capturing sensor %d update (sequential)...\n', 1+s, s);

        sensorUpdate = struct();
        sensorUpdate.sensorIndex = s;
        sensorUpdate.input = struct();
        sensorUpdate.input.objects = captureObjectsData(currentObjects);
        sensorUpdate.input.measurements = measurements{s, t};

        if numel(measurements{s, t})
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

    % IC-LMB has no separate fusion step
    fixtureData.step5_fusion = [];
    finalObjects = currentObjects;

else
    % AA/GA/PU: Parallel processing
    measurementUpdatedDistributions = cell(1, model.numberOfSensors);

    for s = 1:numberOfSensors
        fprintf('  [%d] Capturing sensor %d update (parallel)...\n', 1+s, s);

        sensorUpdate = struct();
        sensorUpdate.sensorIndex = s;
        sensorUpdate.input = struct();
        sensorUpdate.input.objects = captureObjectsData(predictedObjects);  % Same input for all sensors
        sensorUpdate.input.measurements = measurements{s, t};

        if numel(measurements{s, t})
            % Association matrices
            [associationMatrices, posteriorParameters] = generateLmbSensorAssociationMatrices(predictedObjects, measurements{s, t}, model, s);

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
            measurementUpdatedDistributions{s} = computePosteriorLmbSpatialDistributions(predictedObjects, r, W, posteriorParameters, model);

            sensorUpdate.output = struct();
            sensorUpdate.output.updated_objects = captureObjectsData(measurementUpdatedDistributions{s});
        else
            % No measurements - missed detection update
            measurementUpdatedDistributions{s} = predictedObjects;
            for i = 1:numel(predictedObjects)
                measurementUpdatedDistributions{s}(i).r = (measurementUpdatedDistributions{s}(i).r * (1 - model.detectionProbability(s))) / (1 - measurementUpdatedDistributions{s}(i).r * model.detectionProbability(s));
            end

            sensorUpdate.output = struct();
            sensorUpdate.output.updated_objects = captureObjectsData(measurementUpdatedDistributions{s});
            sensorUpdate.association = [];
            sensorUpdate.dataAssociation = [];
        end

        fixtureData.sensorUpdates{s} = sensorUpdate;
    end

    %% Step 5: Fusion
    fprintf('  [%d] Capturing fusion step (%s)...\n', 1+numberOfSensors+1, variant);

    fusionInput = struct();
    fusionInput.per_sensor_objects = cell(1, numberOfSensors);
    for s = 1:numberOfSensors
        fusionInput.per_sensor_objects{s} = captureObjectsData(measurementUpdatedDistributions{s});
    end
    fusionInput.predicted_objects = captureObjectsData(predictedObjects);  % PU needs this

    % Apply fusion
    if strcmp(variant, 'AA')
        fusedObjects = aaLmbTrackMerging(measurementUpdatedDistributions, model);
    elseif strcmp(variant, 'GA')
        fusedObjects = gaLmbTrackMerging(measurementUpdatedDistributions, model);
    else  % PU
        fusedObjects = puLmbTrackMerging(measurementUpdatedDistributions, predictedObjects, model);
    end

    fusionOutput = struct();
    fusionOutput.fused_objects = captureObjectsData(fusedObjects);

    fixtureData.step5_fusion = struct();
    fixtureData.step5_fusion.fusion_type = variant;
    fixtureData.step5_fusion.input = fusionInput;
    fixtureData.step5_fusion.output = fusionOutput;

    finalObjects = fusedObjects;
end

%% Step final: Cardinality estimation
fprintf('  [%d] Capturing cardinality estimation...\n', 1+numberOfSensors+2);

cardinalityInput = struct();
cardinalityInput.existence_probs = [finalObjects.r];

[nMap, mapIndices] = lmbMapCardinalityEstimate([finalObjects.r]);

cardinalityOutput = struct();
cardinalityOutput.n_estimated = nMap;
cardinalityOutput.map_indices = mapIndices;

fixtureData.stepFinal_cardinality = struct();
fixtureData.stepFinal_cardinality.input = cardinalityInput;
fixtureData.stepFinal_cardinality.output = cardinalityOutput;

%% Save to JSON file
% Output to Rust tests directory
rustDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'multisensor-lmb-filters-rs', 'tests', 'data', 'step_by_step');
if ~exist(rustDir, 'dir')
    mkdir(rustDir);
    fprintf('  Created directory: %s\n', rustDir);
end

filename = fullfile(rustDir, sprintf('%s_lmb_step_by_step_seed42.json', lower(variant)));
fprintf('  Saving to: %s\n', filename);

jsonStr = jsonencode(fixtureData);
fid = fopen(filename, 'w');
if fid == -1
    error('Failed to open file: %s', filename);
end
fprintf(fid, '%s', jsonStr);
fclose(fid);

fprintf('\n  %s-LMB step-by-step fixture complete\n', variant);
fprintf('    - Timestep: %d\n', t);
fprintf('    - Sensors: %d\n', numberOfSensors);
fprintf('    - Objects (predicted): %d\n', numel(predictedObjects));
fprintf('    - Objects (final): %d\n', numel(finalObjects));

end  % End of generateVariantFixture

%% =========================================================================
%% Helper Functions
%% =========================================================================

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
    modelData.aaSensorWeights = model.aaSensorWeights;
    modelData.gaSensorWeights = model.gaSensorWeights;
end

function objData = captureObjectsData(objects)
    n = numel(objects);
    objData = cell(1, n);
    for i = 1:n
        obj = struct();
        obj.r = objects(i).r;
        obj.birthTime = objects(i).birthTime;
        obj.birthLocation = objects(i).birthLocation;
        obj.numberOfGmComponents = objects(i).numberOfGmComponents;
        obj.w = objects(i).w;
        obj.mu = objects(i).mu;
        obj.Sigma = objects(i).Sigma;
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
