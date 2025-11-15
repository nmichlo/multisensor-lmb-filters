% Debug Gibbs frequency sampling
addpath('common');
addpath('lmb');

rng = SimpleRng(42);

% Create simple test
n = 2;
m = 2;

associationMatrices.P = [0.7, 0.3; 0.4, 0.6];
associationMatrices.L = [0.05, 0.8, 0.15; 0.05, 0.3, 0.65];
associationMatrices.R = [0.05, 1.0, 1.0; 0.05, 1.0, 1.0];
associationMatrices.C = [1.0, 3.0; 3.0, 1.0];

% Initialize
[v, w] = initialiseGibbsAssociationVectors(associationMatrices.C);
fprintf('Initial v: [%d, %d]\n\n', v(1), v(2));

% Run first few iterations manually
eta = (1:n)';
Sigma = zeros(n, m+1);

fprintf('First 5 iterations:\n');
for i = 1:5
    fprintf('Iter %d: v = [%d, %d], ', i, v(1), v(2));

    % Tally
    ell = n * v + eta;
    fprintf('ell = [%d, %d], ', ell(1), ell(2));
    Sigma(ell) = Sigma(ell) + 1.0;

    % Generate new sample
    [rng, v, w] = generateGibbsSample(rng, associationMatrices.P, v, w);
    fprintf('next v = [%d, %d]\n', v(1), v(2));
end

fprintf('\nSigma after 5 iterations:\n');
disp(Sigma);
