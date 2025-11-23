%% Extract Sigma matrix at t=65 for debugging
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
            fprintf('=== MATLAB t=65 (0-indexed t=64) ===\n');
            fprintf('Number of objects before gating: %d\n', length(objects));
            fprintf('Number of measurements: %d\n', length(measurements{t}));

            % Re-run Murty's core computation to extract Sigma
            n = size(associationMatrices.C, 1);
            m = size(associationMatrices.C, 2);

            % Run Murty's algorithm
            V_internal = murtysAlgorithmWrapper(associationMatrices.C, model.numberOfAssignments);

            % Determine marginal distributions (from lmbMurtysAlgorithm.m lines 30-33)
            W_internal = repmat(V_internal, 1, 1, m+1) == reshape(0:m, 1, 1, m+1);
            J = reshape(associationMatrices.L(n * V_internal + (1:n)), size(V_internal, 1), n);
            L = permute(sum(prod(J, 2) .* W_internal, 1), [2 1 3]);
            Sigma = reshape(L, n, m+1);

            fprintf('\n=== Sigma matrix (first 5 objects, all measurements) ===\n');
            for i = 1:min(5, n)
                fprintf('Sigma[%d,:] = [', i-1);
                for j = 1:(m+1)
                    fprintf('%.17g', Sigma(i, j));
                    if j < m+1
                        fprintf(', ');
                    end
                end
                fprintf(']\n');
            end

            fprintf('\n=== R matrix (first 5 objects) ===\n');
            for i = 1:min(5, n)
                fprintf('R[%d,:] = [', i-1);
                for j = 1:(m+1)
                    fprintf('%.17g', associationMatrices.R(i, j));
                    if j < m+1
                        fprintf(', ');
                    end
                end
                fprintf(']\n');
            end

            fprintf('\n=== Tau = (Sigma .* R) ./ sum(Sigma, 2) ===\n');
            Tau = (Sigma .* associationMatrices.R) ./ sum(Sigma, 2);
            for i = 1:min(5, n)
                fprintf('Tau[%d,:] = [', i-1);
                for j = 1:(m+1)
                    fprintf('%.17g', Tau(i, j));
                    if j < m+1
                        fprintf(', ');
                    end
                end
                fprintf(']\n');
            end

            fprintf('\n=== r = sum(Tau, 2) ===\n');
            r_internal = sum(Tau, 2);
            for i = 1:min(5, n)
                fprintf('r[%d] = %.17g (r == 1.0: %d)\n', i-1, r_internal(i), r_internal(i) == 1.0);
            end
        end

        objects = computePosteriorLmbSpatialDistributions(objects, r, W, posteriorParameters, model);
    else
        objects = updateNoMeasurements(objects, model.detectionProbability);
    end

    % Gate
    objectsLikelyToExist = [objects.r] > model.existenceThreshold;
    objects = objects(objectsLikelyToExist);
end
