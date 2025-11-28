% DEBUG Gibbs RNG sequence
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

fprintf('=== MATLAB GIBBS RNG TRACE ===\n');
fprintf('Dimensions: %s\n', mat2str(size(L)));
fprintf('m values: [%d %d %d]\n', size(L, 1)-1, size(L, 2)-1, size(L, 3)-1);

% Manually trace first Gibbs sample
ell = size(L);
m = ell(1:end-1) - 1;
n = ell(end);
numberOfSensors = length(m);
V = zeros(n, numberOfSensors);
W = zeros(max(m), numberOfSensors);

fprintf('\nFirst Gibbs sample RNG calls:\n');
rngCallCount = 0;

% For each sensor
for s = 1:numberOfSensors
    % For each object
    for i = 1:n
        k = V(i, s) + (V(i, s) == 0);
        u = [(V(i, :) + 1), i];

        % For each measurement
        for j = k:m(s)
            if ((W(j, s) == 0) || (W(j, s) == i))
                % Detection
                u(s) = j + 1;
                q_val = u(1);
                pi_val = 1;
                for idx = 2:length(u)
                    pi_val = pi_val * ell(idx-1);
                    q_val = q_val + pi_val * (u(idx) - 1);
                end
                q = round(q_val);

                % Miss
                u(s) = 1;
                r_val = u(1);
                pi_val = 1;
                for idx = 2:length(u)
                    pi_val = pi_val * ell(idx-1);
                    r_val = r_val + pi_val * (u(idx) - 1);
                end
                r = round(r_val);

                % Sample probability
                P = 1 / (exp(L(r) - L(q)) + 1);

                % Get random value
                [filterRng, sample] = filterRng.rand();
                rngCallCount = rngCallCount + 1;

                if rngCallCount <= 10
                    fprintf('  RNG call %d: s=%d, i=%d, j=%d, rand=%.6f, p=%.6f, q_idx=%d, r_idx=%d, L[q]=%.6f, L[r]=%.6f, diff=%.6f\n', ...
                        rngCallCount, s, i, j, sample, P, q, r, L(q), L(r), L(r) - L(q));
                end

                if (sample < P)
                    V(i, s) = j;
                    W(j, s) = i;
                    break;
                else
                    V(i, s) = 0;
                    W(j, s) = 0;
                end
            end
        end
    end
end

fprintf('\nFirst sample V (reshaped): [');
fprintf('%d ', reshape(V, 1, n * numberOfSensors));
fprintf(']\n');
