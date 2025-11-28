% DEBUG likelihood computation details
clc; close all;
setPath;

% Setup (matching test exactly)
seed = 1;
numberOfSensors = 3;
clutterRates = [5 5 5];
detectionProbabilities = [0.67 0.70 0.73];
q = [4 3 2];

% Generate model (seed 0)
modelRng = SimpleRng(0);
[modelRng, model] = generateMultisensorModel(modelRng, numberOfSensors, clutterRates, detectionProbabilities, q, 'PU', 'LBP', 'Fixed');

% Generate ground truth (seed 1)
trialRng = SimpleRng(seed);
[trialRng, groundTruth, measurements, groundTruthRfs] = generateMultisensorGroundTruth(trialRng, model);

% Truncate to 10 timesteps
measurementsShort = measurements(:, 1:10);

% Initialize filter (seed 1001)
filterRng = SimpleRng(seed + 1000);
hypotheses = model.hypotheses;

% Process t=1
t = 1;
[model.birthTrajectory.birthTime] = deal(t);
objects(1:model.numberOfBirthLocations) = model.birthTrajectory;

% Prediction
priorHypothesis = lmbmPredictionStep(hypotheses(1), model, t);

fprintf('=== MATLAB LIKELIHOOD DEBUG ===\n');
fprintf('model.detectionProbability: %s\n', mat2str(model.detectionProbability));
fprintf('model.clutterPerUnitVolume: %s\n', mat2str(model.clutterPerUnitVolume));

% Manually compute first few likelihoods (mimicking generateMultisensorLmbmAssociationMatrices)
dimensions = [zeros(1, model.numberOfSensors) numel(priorHypothesis.r)];
for s = 1:model.numberOfSensors
    dimensions(s) = numel(measurementsShort{s, t}) + 1;
end
numberOfEntries = prod(dimensions);
pageSizes = [1 cumprod(dimensions(1:end-1))];

fprintf('\nDimensions: %s\n', mat2str(dimensions));

debugCount = 0;
maxDebug = 3;

for ell = 1:numberOfEntries
    % Get association vector
    n = ell;
    m = numel(pageSizes);
    u = zeros(1, m);
    for i_loop = 1:m
        j = m - i_loop + 1;
        zeta = floor(n / pageSizes(j));
        eta = mod(n, pageSizes(j));
        u(j) = zeta + (eta ~= 0);
        n = n - pageSizes(j) * (zeta - (eta == 0));
    end

    i = u(end);
    a = u(1:end-1) - 1;

    % Check if we have assignments
    assignments = a > 0;
    numberOfAssignments = sum(assignments);

    if (numberOfAssignments > 0 && debugCount < maxDebug)
        % Determine measurement vector
        z = zeros(model.zDimension * numberOfAssignments, 1);
        C = zeros(model.zDimension * numberOfAssignments, model.xDimension);
        counter = 0;
        for s = 1:model.numberOfSensors
            if (assignments(s))
                start = model.zDimension * counter + 1;
                finish = start + model.zDimension - 1;
                z(start:finish) = measurementsShort{s, t}{a(s)};
                C(start:finish, :) = model.C{s};
                counter = counter + 1;
            end
        end

        % Likelihood
        Q = blkdiag(model.Q{assignments});
        nu = z - C * priorHypothesis.mu{i};
        Z = C * priorHypothesis.Sigma{i} * C' + Q;

        fprintf('  Z matrix shape: %dx%d\n', size(Z, 1), size(Z, 2));
        fprintf('  Q matrix shape: %dx%d\n', size(Q, 1), size(Q, 2));
        fprintf('  Z det: %.6f\n', det(Z));
        fprintf('  2*pi*Z det: %.6f\n', det(2*pi*Z));

        ZInv = inv(Z);
        K = priorHypothesis.Sigma{i} * C' * ZInv;
        eta = -0.5 * log(det(2*pi*Z));
        Pd = sum(log(model.detectionProbability(assignments))) + sum(log(1 - model.detectionProbability(~assignments)));
        kappa = sum(log(model.clutterPerUnitVolume(assignments)));
        mahalanobis = 0.5 * nu' * ZInv * nu;
        L = log(priorHypothesis.r(i)) + Pd + eta - mahalanobis - kappa;

        fprintf('\n=== LIKELIHOOD DEBUG #%d ===\n', debugCount + 1);
        fprintf('  Linear index ell: %d\n', ell);
        fprintf('  Object i: %d\n', i);
        fprintf('  Assignments: %s\n', mat2str(assignments));
        fprintf('  hypothesis.r(i): %.6f\n', priorHypothesis.r(i));
        fprintf('  log(r): %.6f\n', log(priorHypothesis.r(i)));
        fprintf('  pd_log: %.6f\n', Pd);
        fprintf('  eta: %.6f\n', eta);
        fprintf('  mahalanobis: %.6f\n', mahalanobis);
        fprintf('  kappa_log: %.6f\n', kappa);
        fprintf('  L = %.6f\n', L);

        debugCount = debugCount + 1;
    end
end
