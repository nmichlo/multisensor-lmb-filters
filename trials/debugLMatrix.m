% DEBUG L matrix values
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

% Generate association matrices
[L, posteriorParameters] = generateMultisensorLmbmAssociationMatrices(priorHypothesis, measurementsShort(:, t), model);

fprintf('=== MATLAB L MATRIX DEBUG ===\n');
fprintf('Dimensions: %s\n', mat2str(size(L)));
fprintf('Total entries: %d\n', numel(L));

% Print first 20 L values
fprintf('\nFirst 20 L values (linear indexing):\n');
for ell = 1:min(20, numel(L))
    fprintf('  L[%d] = %.6f\n', ell, L(ell));
end

% Print specific indices from Gibbs trace
% From MATLAB Gibbs trace: q_idx=2, r_idx=1
fprintf('\nSpecific indices from Gibbs trace:\n');
fprintf('  L[1] = %.6f\n', L(1));
fprintf('  L[2] = %.6f\n', L(2));
