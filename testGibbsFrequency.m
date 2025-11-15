% Test frequency-based Gibbs sampling cross-language equivalence
addpath('common');
addpath('lmb');

fprintf('Testing Gibbs Frequency Sampling Cross-Language Equivalence\n');
fprintf('============================================================\n\n');

% Set up test scenario
rng = SimpleRng(42);

% Create simple association matrices
n = 2;  % 2 objects
m = 2;  % 2 measurements

% P matrix (sampling probabilities)
associationMatrices.P = [0.7, 0.3;
                         0.4, 0.6];

% L matrix [eta, L1, L2] (likelihoods)
associationMatrices.L = [0.05, 0.8, 0.15;
                         0.05, 0.3, 0.65];

% R matrix [phi/eta, 1, 1] (existence ratios)
associationMatrices.R = [0.05, 1.0, 1.0;
                         0.05, 1.0, 1.0];

% C matrix (cost for initialization)
associationMatrices.C = [1.0, 3.0;
                         3.0, 1.0];

% Run frequency-based Gibbs sampling
numberOfSamples = 1000;
[rng, r, W] = lmbGibbsFrequencySampling(rng, associationMatrices, numberOfSamples);

fprintf('Results from frequency-based Gibbs sampling (%d samples):\n\n', numberOfSamples);

fprintf('Existence probabilities (r):\n');
fprintf('  Object 1: %.10f\n', r(1));
fprintf('  Object 2: %.10f\n\n', r(2));

fprintf('Marginal association probabilities (W):\n');
fprintf('  Object 1: [%.6f, %.6f, %.6f] (sum=%.6f)\n', W(1,1), W(1,2), W(1,3), sum(W(1,:)));
fprintf('  Object 2: [%.6f, %.6f, %.6f] (sum=%.6f)\n', W(2,1), W(2,2), W(2,3), sum(W(2,:)));

fprintf('\nVerification:\n');
fprintf('  All W rows sum to 1: %s\n', mat2str(abs(sum(W,2) - 1) < 1e-10));
fprintf('  All r in [0,1]: %s\n', mat2str(all(r >= 0 & r <= 1)));
