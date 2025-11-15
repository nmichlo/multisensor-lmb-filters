function [rng, r, W] = lmbGibbsFrequencySampling(rng, associationMatrices, numberOfSamples)
% LMBGIBBSFREQUENCYSAMPLING -- Determine posterior existence probabilities and association weights using a Gibbs sampler
%  [rng, r, W] = lmbGibbsFrequencySampling(rng, associationMatrices, numberOfSamples)
%
%   This function determines each object's posterior existence and marginal
%   association probabilities using Gibbs sampling. This functions uses the
%   frequencies of the samples to determine the marginals. It does not
%   reject any samples, unlike lbmGibbsSampling. However, it is a bit
%   slower in Matlab, but perhaps faster in languages with faster for
%   loops.
%
%   See also runLmbFilter, generateLmbAssociationMatrices,
%   computePosteriorLmbSpatialDistributions, lmbMurtysAlgorithm, loopyBeliefPropagation
%
%   Inputs
%       rng - SimpleRng object. Random number generator.
%       associationMatrices - struct. A struct whose fields are the arrays required
%           by the various data association algorithms.
%       numberOfSamples - double. The number of Gibbs samples we want to
%           generate.
%
%   Output
%       rng - SimpleRng object. Updated random number generator state.
%       r - array. Each object's posterior existence probability.
%       W - array. An array of marginal association probabilities, where
%           each row is an object's marginal association probabilities.

%% Declare variables
[n, m] = size(associationMatrices.P);
% Association vectors
[v, w] = initialiseGibbsAssociationVectors(associationMatrices.C);
% Marginals
eta = (1:n)';
Sigma = zeros(n, m+1);
%% Gibbs sampling
for i = 1:numberOfSamples
    %% Add up tally
    ell = n * v + eta;
    Sigma(ell) = Sigma(ell) + (1 / numberOfSamples);
    %% Generate a new Gibbs sample
    [rng, v, w] = generateGibbsSample(rng, associationMatrices.P, v, w);
end
%% Normalise
Tau = Sigma .* associationMatrices.R;
%% Determine existence probabilities
r = sum(Tau, 2);
%% Determine marginal association probabilities
W =  Tau ./ r;
end