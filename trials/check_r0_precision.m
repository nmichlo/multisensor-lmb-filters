%% Check r[0] precision at t=65
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
        objects = computePosteriorLmbSpatialDistributions(objects, r, W, posteriorParameters, model);
    else
        objects = updateNoMeasurements(objects, model.detectionProbability);
    end

    % Gate
    objectsLikelyToExist = [objects.r] > model.existenceThreshold;
    objects = objects(objectsLikelyToExist);

    if t == 65
        fprintf('=== t=65 (0-indexed t=64) AFTER GATING ===\n');
        r_vals = [objects.r];

        fprintf('\nHigh precision r values:\n');
        for i = 1:min(8, length(r_vals))
            fprintf('r[%d] = %.17g\n', i-1, r_vals(i));
            fprintf('  (r == 1.0): %d\n', r_vals(i) == 1.0);
            fprintf('  (r >= 1.0): %d\n', r_vals(i) >= 1.0);
        end

        % Check after adjustment
        r_adj = r_vals - 1e-6;
        fprintf('\nAfter r = r - 1e-6:\n');
        for i = 1:min(8, length(r_adj))
            fprintf('r_adj[%d] = %.17g\n', i-1, r_adj(i));
        end

        % Sort and show order
        [~, sortedIndices] = sort(-r_adj);
        fprintf('\nSorted indices (descending r_adj): [');
        for i = 1:min(9, length(sortedIndices))
            fprintf('%d', sortedIndices(i)-1);
            if i < min(9, length(sortedIndices))
                fprintf(', ');
            end
        end
        fprintf(']\n');
    end
end
