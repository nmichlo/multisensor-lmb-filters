%% Check Murty output at t=65
clc;
pkg load statistics;
setPath;

modelRng = SimpleRng(0);
[modelRng, model] = generateModel(modelRng, 2.0, 0.95, 'Murty', 'Fixed');

trialRng = SimpleRng(42);
[trialRng, groundTruth, measurements] = generateGroundTruth(trialRng, model);

objects = model.object;
for t = 1:65
    objects = lmbPredictionStep(objects, model, t);

    if ~isempty(measurements{t})
        [associationMatrices, posteriorParameters] = generateLmbAssociationMatrices(objects, measurements{t}, model);
        [r, W, V] = lmbMurtysAlgorithm(associationMatrices, model.numberOfAssignments);

        if t == 65
            fprintf('=== t=65 MURTY OUTPUT ===\n');
            for i = 1:min(8, length(r))
                fprintf('r[%d] = %.17g\n', i-1, r(i));
                fprintf('  (r == 1.0): %d\n', r(i) == 1.0);
                fprintf('  (r > 0.999999): %d\n', r(i) > 0.999999);
            end
        end

        objects = computePosteriorLmbSpatialDistributions(objects, r, W, posteriorParameters, model);

        if t == 65
            fprintf('\n=== t=65 AFTER POSTERIOR UPDATE ===\n');
            for i = 1:min(8, length(objects))
                fprintf('r[%d] = %.17g\n', i-1, objects(i).r);
                fprintf('  (r == 1.0): %d\n', objects(i).r == 1.0);
            end
        end
    else
        objects = updateNoMeasurements(objects, model.detectionProbability);
    end

    objectsLikelyToExist = [objects.r] > model.existenceThreshold;
    objects = objects(objectsLikelyToExist);
end
