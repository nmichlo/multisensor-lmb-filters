%% Debug script for seed 42, timestep 64 - LMB-Murty divergence point
clc; close all;
pkg load statistics;
setPath;

%% Generate model (seed 0)
modelRng = SimpleRng(0);
[modelRng, model] = generateModel(modelRng, 2.0, 0.95, 'Murty', 'Fixed');
model.dataAssociationMethod = 'Murty';

%% Generate ground truth (seed 42)
trialRng = SimpleRng(42);
[trialRng, groundTruth, measurements] = generateGroundTruth(trialRng, model);

%% Run filter up to t=63
filterRng = SimpleRng(42 + 1000);
objects = model.object;

fprintf('\n=== Running LMB-Murty for seed 42 up to t=63 ===\n');

for t = 1:64
    % Prediction
    objects = lmbPredictionStep(objects, model, t);

    % Measurement update
    if ~isempty(measurements{t})
        % Generate association matrices
        [associationMatrices, posteriorParameters] = generateLmbAssociationMatrices(objects, measurements{t}, model);

        % Murty's algorithm
        [r, W, V] = lmbMurtysAlgorithm(associationMatrices, model.numberOfAssignments);

        % Log details at t=63
        if t == 63
            fprintf('\n=== Timestep 63 (before failure) ===\n');
            fprintf('Number of objects: %d\n', length(objects));
            fprintf('Number of measurements: %d\n', length(measurements{t}));
            fprintf('\nExistence probabilities r:\n');
            for i = 1:length(r)
                if r(i) > 0.01
                    fprintf('  Object %d: r = %.6f\n', i, r(i));
                end
            end
        end

        % Compute posterior
        objects = computePosteriorLmbSpatialDistributions(objects, r, W, posteriorParameters, model);

        % Log after update at t=63
        if t == 63
            fprintf('\n=== After posterior update at t=63 ===\n');
            for i = 1:length(objects)
                if objects(i).r > 0.01
                    fprintf('Object %d: r=%.6f, num_comp=%d, mu[1]=%.4f\n', ...
                            i, objects(i).r, objects(i).numberOfGmComponents, objects(i).mu{1}(1));
                end
            end
        end
    else
        % No measurements
        objects = updateNoMeasurements(objects, model.detectionProbability);
    end

    % Gate objects
    objectsLikelyToExist = [objects.r] > model.existenceThreshold;
    objects = objects(objectsLikelyToExist);

    if t == 63
        fprintf('\n=== After gating at t=63 ===\n');
        fprintf('Objects remaining: %d\n', length(objects));
    end
end

%% Now run timestep 64 with detailed logging
fprintf('\n\n=== CRITICAL TIMESTEP 64 (FAILURE POINT) ===\n');
t = 64;

% Prediction
fprintf('\n1. Prediction step\n');
objects = lmbPredictionStep(objects, model, t);
fprintf('   Objects after prediction: %d\n', length(objects));

% Measurement update
fprintf('\n2. Generating association matrices\n');
fprintf('   Measurements at t=64: %d\n', length(measurements{t}));

[associationMatrices, posteriorParameters] = generateLmbAssociationMatrices(objects, measurements{t}, model);

fprintf('   Association matrix L dimensions: %d × %d\n', size(associationMatrices.L));
fprintf('   Cost matrix C dimensions: %d × %d\n', size(associationMatrices.C));

fprintf('\n3. Running Murty''s algorithm (K=%d)\n', model.numberOfAssignments);
[r, W, V] = lmbMurtysAlgorithm(associationMatrices, model.numberOfAssignments);

fprintf('\n4. Murty results:\n');
fprintf('   Existence probabilities r:\n');
for i = 1:length(r)
    if r(i) > 0.01
        fprintf('     Object %d: r = %.6f\n', i, r(i));
    end
end

fprintf('\n   Association weights W (objects with r > 0.01):\n');
for i = 1:length(r)
    if r(i) > 0.01
        fprintf('     Object %d: [', i);
        nShow = min(10, size(W, 2));
        for j = 1:nShow
            fprintf('%.4f', W(i, j));
            if j < nShow
                fprintf(', ');
            end
        end
        if size(W, 2) > 10
            fprintf(', ...');
        end
        fprintf(']\n');
    end
end

fprintf('\n   K-best assignments V (first 10 rows):\n');
nShow = min(10, size(V, 1));
for i = 1:nShow
    fprintf('     Event %d: [', i);
    nCols = min(10, size(V, 2));
    for j = 1:nCols
        fprintf('%d', V(i, j));
        if j < nCols
            fprintf(', ');
        end
    end
    if size(V, 2) > 10
        fprintf(', ...');
    end
    fprintf(']\n');
end

fprintf('\n5. Computing posterior spatial distributions\n');
objects = computePosteriorLmbSpatialDistributions(objects, r, W, posteriorParameters, model);

fprintf('\n6. State estimates after update:\n');
for i = 1:length(objects)
    if objects(i).r > 0.01
        fprintf('   Object %d: r=%.6f, num_comp=%d\n', i, objects(i).r, objects(i).numberOfGmComponents);
        fprintf('     mu{1}: [%.4f, %.4f, %.4f, %.4f]\n', ...
                objects(i).mu{1}(1), objects(i).mu{1}(2), objects(i).mu{1}(3), objects(i).mu{1}(4));
    end
end

% Extract MAP estimate
[nMap, mapIndices] = lmbMapCardinalityEstimate([objects.r]);

fprintf('\n7. MAP cardinality estimate:\n');
fprintf('   Estimated cardinality: %d\n', nMap);
fprintf('   Selected objects: [');
for i = 1:length(mapIndices)
    fprintf('%d', mapIndices(i));
    if i < length(mapIndices)
        fprintf(', ');
    end
end
fprintf(']\n');

fprintf('\n8. Final state estimate mu:\n');
for i = 1:length(mapIndices)
    objIdx = mapIndices(i);
    fprintf('   Target %d (object %d): mu{1} = [%.6f, %.6f, %.6f, %.6f]\n', ...
            i-1, objIdx, ...  % i-1 for 0-indexed target numbering
            objects(objIdx).mu{1}(1), objects(objIdx).mu{1}(2), ...
            objects(objIdx).mu{1}(3), objects(objIdx).mu{1}(4));
end

fprintf('\n=== Debug script complete ===\n');
fprintf('\nCompare these values with Rust debug output to identify discrepancy source\n');
