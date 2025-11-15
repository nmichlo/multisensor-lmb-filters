% GENERATESINGLE SENSORACCURACYFIXTURES_QUICK -- Quick validation fixture (seed 42)
%
% This script generates a quick validation fixture for seed 42 with:
% - LMB filters: all 100 timesteps
% - LMBM filters: only 10 timesteps (100x faster, validates core mechanics)
%
% Output: fixtures/accuracy/single_trial_42.json

%% Admin
clc; close all;
pkg load statistics; % For Octave
setPath;

%% Configuration
seed = 42;
simulationLength = 100;  % Full simulation for ground truth and LMB
lmbmSimulationLength = 10;  % Reduced for LMBM (100x faster)
lmbDataAssociationMethods = {'LBP', 'Gibbs', 'Murty'};
numberOfLmbAssociationMethods = numel(lmbDataAssociationMethods);
lmbmDataAssociationMethods = {'Gibbs', 'Murty'};
numberOfLmbmAssociationMethods = numel(lmbmDataAssociationMethods);

%% Generate model (same for all trials, fixed seed 0)
fprintf('Generating model...\n');
modelRng = SimpleRng(0);
[modelRng, model] = generateModel(modelRng, 2, 0.95, 'LBP');

%% Create fixtures directory
fixturesDir = fullfile(fileparts(mfilename('fullpath')), '..', 'fixtures', 'accuracy');
if ~exist(fixturesDir, 'dir')
    mkdir(fixturesDir);
    fprintf('Created directory: %s\n', fixturesDir);
end

%% Generate fixture for seed 42
startTime = tic;
fprintf('\n=== Generating quick validation fixture for seed %d ===\n', seed);

% Generate ground truth for this seed (full 100 timesteps)
fprintf('  [%s] Generating ground truth...\n', datestr(now, 'HH:MM:SS'));
trialRng = SimpleRng(seed);
[trialRng, groundTruth, measurements, groundTruthRfs] = generateGroundTruth(trialRng, model);
fprintf('  [%s] Ground truth generated (%d timesteps, %.1fs)\n', datestr(now, 'HH:MM:SS'), numel(measurements), toc(startTime));

% Initialize storage
eOspaLmb = cell(1, numberOfLmbAssociationMethods);
hOspaLmb = cell(1, numberOfLmbAssociationMethods);
lmbCardinality = cell(1, numberOfLmbAssociationMethods);
eOspaLmbm = cell(1, numberOfLmbmAssociationMethods);
hOspaLmbm = cell(1, numberOfLmbmAssociationMethods);
lmbmCardinality = cell(1, numberOfLmbmAssociationMethods);

%% Run LMB filters (all 100 timesteps)
for i = 1:numberOfLmbAssociationMethods
    filterStartTime = tic;
    fprintf('  [%s] Running LMB-%s (%d timesteps)...', datestr(now, 'HH:MM:SS'), lmbDataAssociationMethods{i}, simulationLength);
    fflush(stdout);
    model.dataAssociationMethod = lmbDataAssociationMethods{i};
    filterRng = SimpleRng(seed + 1000);
    [filterRng, stateEstimates] = runLmbFilter(filterRng, model, measurements);
    [eOspaLmb{i}, hOspaLmb{i}, lmbCardinality{i}] = computeSimulationOspa(model, groundTruthRfs, stateEstimates);
    fprintf(' done (%.1fs)\n', toc(filterStartTime));
end

%% Run LMBM filters (only first 10 timesteps)
for i = 1:numberOfLmbmAssociationMethods
    filterStartTime = tic;
    fprintf('  [%s] Running LMBM-%s (%d timesteps, REDUCED)...', datestr(now, 'HH:MM:SS'), lmbmDataAssociationMethods{i}, lmbmSimulationLength);
    fflush(stdout);
    model.dataAssociationMethod = lmbmDataAssociationMethods{i};
    filterRng = SimpleRng(seed + 1000);

    % Run filter with only first 10 measurements
    measurementsShort = measurements(1:lmbmSimulationLength);
    [filterRng, stateEstimates] = runLmbmFilter(filterRng, model, measurementsShort);

    % Create shortened ground truth RFS for OSPA computation
    groundTruthRfsShort = struct();
    groundTruthRfsShort.x = groundTruthRfs.x(1:lmbmSimulationLength);
    groundTruthRfsShort.mu = groundTruthRfs.mu(1:lmbmSimulationLength);
    groundTruthRfsShort.Sigma = groundTruthRfs.Sigma(1:lmbmSimulationLength);

    [eOspaLmbm{i}, hOspaLmbm{i}, lmbmCardinality{i}] = computeSimulationOspa(model, groundTruthRfsShort, stateEstimates);
    fprintf(' done (%.1fs)\n', toc(filterStartTime));
end

%% Build JSON structure
fixtureData = struct();
fixtureData.seed = seed;
fixtureData.filterVariants = {};

% Add LMB filter results (100 timesteps each)
for i = 1:numberOfLmbAssociationMethods
    variant = struct();
    variant.name = ['LMB-' lmbDataAssociationMethods{i}];
    variant.eOspa = eOspaLmb{i};
    variant.hOspa = hOspaLmb{i};
    variant.cardinality = lmbCardinality{i};
    fixtureData.filterVariants{end+1} = variant;
end

% Add LMBM filter results (10 timesteps each)
for i = 1:numberOfLmbmAssociationMethods
    variant = struct();
    variant.name = ['LMBM-' lmbmDataAssociationMethods{i}];
    variant.eOspa = eOspaLmbm{i};
    variant.hOspa = hOspaLmbm{i};
    variant.cardinality = lmbmCardinality{i};
    fixtureData.filterVariants{end+1} = variant;
end

%% Save to JSON file
filename = fullfile(fixturesDir, sprintf('single_trial_%d.json', seed));
fprintf('  [%s] Saving to: %s\n', datestr(now, 'HH:MM:SS'), filename);

% Convert to JSON string
jsonStr = jsonencode(fixtureData);

% Write to file
fid = fopen(filename, 'w');
if fid == -1
    error('Failed to open file: %s', filename);
end
fprintf(fid, '%s', jsonStr);
fclose(fid);

totalElapsedTime = toc(startTime);
fprintf('  [%s] ✓ Quick validation fixture complete (%.1fs total)\n', datestr(now, 'HH:MM:SS'), totalElapsedTime);
fprintf('\nOutput: %s\n', filename);
fprintf('  - LMB variants: 3 × 100 timesteps\n');
fprintf('  - LMBM variants: 2 × 10 timesteps (reduced)\n');
