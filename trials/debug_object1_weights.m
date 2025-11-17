% Debug Object 1 weight pruning
% Load the fixture to get exact values
fixture = jsondecode(fileread('../fixtures/step_by_step/lmb_step_by_step_seed42.json'));

% Get Object 1 data
W = fixture.step4_update.input.W(2,:);  % Row 2 = Object 1 (1-indexed)
PP_w = fixture.step4_update.input.posteriorParameters(2).w;  % Object 1 posterior weights

fprintf('W shape: %dx%d\n', size(W, 1), size(W, 2));
fprintf('PP_w shape: %dx%d\n', size(PP_w, 1), size(PP_w, 2));

% Compute posterior weights (matching MATLAB logic)
numberOfPosteriorComponents = numel(PP_w);
posteriorWeights = reshape(W' .* PP_w, 1, numberOfPosteriorComponents);

fprintf('Total weights: %d\n', numberOfPosteriorComponents);
fprintf('Posterior weights shape after reshape: %dx%d\n', size(posteriorWeights, 1), size(posteriorWeights, 2));

% Normalize
posteriorWeights = posteriorWeights ./ sum(posteriorWeights);

% Sort
[sortedWeights, sortedIndices] = sort(posteriorWeights, 'descend');

% Apply threshold
threshold = 1e-6;
significantComponents = sortedWeights > threshold;
numSignificant = sum(significantComponents);

fprintf('\nWeights above threshold %e: %d\n', threshold, numSignificant);
fprintf('Top 20 weights:\n');
for i = 1:min(20, length(sortedWeights))
    marker = '';
    if significantComponents(i)
        marker = ' ✓';
    end
    fprintf('  [%2d] %.10f%s\n', i, sortedWeights(i), marker);
end

% Now apply the MATLAB filtering logic
significantWeights = sortedWeights(significantComponents);
fprintf('\nAfter filtering to significant components:\n');
fprintf('Number of components: %d\n', length(significantWeights));
fprintf('Significant weights: %s\n', mat2str(significantWeights(1:min(10, end)), 10));

% Check if capped by maximum
maxComponents = 100;
if length(significantWeights) > maxComponents
    fprintf('Would be capped to %d components\n', maxComponents);
end

% Now check what the fixture output has
fprintf('\n=== FIXTURE OUTPUT ===\n');
outputWeights = fixture.step4_update.output.posterior_objects(2).w;
fprintf('Fixture output Object 1 has %d components\n', length(outputWeights));
fprintf('Output weights: %s\n', mat2str(outputWeights', 10));
