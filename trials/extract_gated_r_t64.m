%% Extract r values AFTER gating at t=64
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
        fprintf('Number of objects after gating: %d\n', length(objects));
        fprintf('Existence threshold: %.10f\n', model.existenceThreshold);

        fprintf('\nGated r values (passed to MAP):\n');
        for i = 1:length(objects)
            fprintf('r[%d] = %.15f\n', i-1, objects(i).r);
        end

        % Run MAP and show result
        [nMap, mapIndices] = lmbMapCardinalityEstimate([objects.r]);
        fprintf('\nMAP result:\n');
        fprintf('  n_map = %d\n', nMap);
        fprintf('  indices (0-indexed) = [');
        for i = 1:length(mapIndices)
            fprintf('%d', mapIndices(i)-1);
            if i < length(mapIndices)
                fprintf(', ');
            end
        end
        fprintf(']\n');

        fprintf('\nTarget estimates:\n');
        for i = 1:nMap
            j = mapIndices(i);
            fprintf('  Target %d (object %d): mu[1] = %.10f\n', i-1, j-1, objects(j).mu{1}(1));
        end
    end
end
