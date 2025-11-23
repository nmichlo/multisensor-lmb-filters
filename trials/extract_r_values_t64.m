%% Extract exact r values at t=64 for seed 42
clc;
pkg load statistics;
setPath;

modelRng = SimpleRng(0);
[modelRng, model] = generateModel(modelRng, 2.0, 0.95, 'Murty', 'Fixed');
model.dataAssociationMethod = 'Murty';

trialRng = SimpleRng(42);
[trialRng, groundTruth, measurements] = generateGroundTruth(trialRng, model);

% Run filter up to t=64
objects = model.object;
for t = 1:64
    objects = lmbPredictionStep(objects, model, t);

    if ~isempty(measurements{t})
        [associationMatrices, posteriorParameters] = generateLmbAssociationMatrices(objects, measurements{t}, model);
        [r, W, V] = lmbMurtysAlgorithm(associationMatrices, model.numberOfAssignments);

        if t == 64
            fprintf('=== t=64 MATLAB r values ===\n');
            fprintf('Number of objects: %d\n', length(r));
            fprintf('Number of measurements: %d\n', length(measurements{t}));
            fprintf('\nExact r values (all objects):\n');
            for i = 1:length(r)
                fprintf('r[%d] = %.15f\n', i-1, r(i));  % 0-indexed for comparison
            end

            fprintf('\nNumber with r > 0.01: %d\n', sum(r > 0.01));
            fprintf('Number with r > 0.99: %d\n', sum(r > 0.99));
            fprintf('Number with r == 1.0: %d\n', sum(r == 1.0));
            fprintf('Max r: %.15f\n', max(r));
            fprintf('Min r: %.15f\n', min(r));
        end

        objects = computePosteriorLmbSpatialDistributions(objects, r, W, posteriorParameters, model);
    else
        objects = updateNoMeasurements(objects, model.detectionProbability);
    end

    objectsLikelyToExist = [objects.r] > model.existenceThreshold;
    objects = objects(objectsLikelyToExist);
end
