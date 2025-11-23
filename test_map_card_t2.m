% Test MAP cardinality for AA-LMB t=2 seed 12345
% Add path to common directory
addpath('common');

% Rust r_vec at t=2 after gating
r = [0.5880293, 0.3022491, 0.1019527, 0.3841591, 0.0328940, 0.0857647];

fprintf('Testing MAP cardinality with r = [');
fprintf('%.7f ', r);
fprintf(']\n\n');

% Call MATLAB MAP cardinality function
[nMap, mapIndices] = lmbMapCardinalityEstimate(r);

fprintf('MATLAB result:\n');
fprintf('  nMap = %d\n', nMap);
fprintf('  mapIndices = [');
fprintf('%d ', mapIndices);
fprintf(']\n\n');

% Show selected r values
fprintf('Selected objects:\n');
for i = 1:nMap
    fprintf('  Object %d (index=%d): r=%.7f\n', i, mapIndices(i), r(mapIndices(i)));
end
