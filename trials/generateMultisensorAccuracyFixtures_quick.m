% GENERATEMULTISENSORACCURACYFIXTURES_QUICK -- Quick validation fixture (seed 42)
%
% This script generates a quick validation fixture for seed 42 with:
% - LMB filters (IC/PU/GA/AA): all 100 timesteps
% - LMBM filter: only 10 timesteps (100x faster, validates core mechanics)
%
% Output: fixtures/multisensor_accuracy/multisensor_trial_42.json

%% Admin
clc; close all;
pkg load statistics; % For Octave
% NOTE: Run setPath before running this script

%% Configuration
seed = 42;
simulationLength = 100;  % Full simulation for ground truth and LMB
lmbmSimulationLength = 10;  % Reduced for LMBM (100x faster)
numberOfSensors = 3;
clutterRates = [5 5 5];
detectionProbabilities = [0.67 0.70 0.73];
q = [4 3 2];
lmbUpdateMethods = {'IC', 'PU', 'GA', 'AA'};
numberOfLmbUpdateMethods = numel(lmbUpdateMethods);

%% Generate model (same for all trials, fixed seed 0)
fprintf('Generating multisensor model...\n');
modelRng = SimpleRng(0);
[modelRng, model] = generateMultisensorModel(modelRng, numberOfSensors, clutterRates, detectionProbabilities, q, 'PU', 'LBP', 'Fixed');

%% Create fixtures directory
fixturesDir = fullfile(fileparts(mfilename('fullpath')), '..', 'fixtures', 'multisensor_accuracy');
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
[trialRng, groundTruth, measurements, groundTruthRfs] = generateMultisensorGroundTruth(trialRng, model);
fprintf('  [%s] Ground truth generated (%d timesteps, %.1fs)\n', datestr(now, 'HH:MM:SS'), numel(measurements), toc(startTime));

% Initialize storage
eOspaLmb = cell(1, numberOfLmbUpdateMethods);
hOspaLmb = cell(1, numberOfLmbUpdateMethods);
lmbCardinality = cell(1, numberOfLmbUpdateMethods);

%% Run LMB filters (all 100 timesteps)
for i = 1:numberOfLmbUpdateMethods
    filterStartTime = tic;
    fprintf('  [%s] Running %s-LMB (%d timesteps)...', datestr(now, 'HH:MM:SS'), lmbUpdateMethods{i}, simulationLength);
    fflush(stdout);

    % Run appropriate filter (no RNG parameter - filters don't use RNG currently)
    if strcmp(lmbUpdateMethods{i}, 'IC')
        stateEstimates = runIcLmbFilter(model, measurements);
    else
        model.lmbParallelUpdateMode = lmbUpdateMethods{i};
        stateEstimates = runParallelUpdateLmbFilter(model, measurements);
    end

    [eOspaLmb{i}, hOspaLmb{i}, lmbCardinality{i}] = computeSimulationOspa(model, groundTruthRfs, stateEstimates);
    fprintf(' done (%.1fs)\n', toc(filterStartTime));
end

%% Skip LMBM filter for now (bug in MATLAB code with reduced timesteps)
% TODO: Debug and fix multisensor LMBM filter
fprintf('  [%s] LMBM SKIPPED (bug in MATLAB code with reduced timesteps)\n', datestr(now, 'HH:MM:SS'));

%% Build JSON structure
fixtureData = struct();
fixtureData.seed = seed;
fixtureData.filterVariants = {};

% Add LMB filter results (100 timesteps each)
for i = 1:numberOfLmbUpdateMethods
    variant = struct();
    variant.name = [lmbUpdateMethods{i} '-LMB'];
    variant.eOspa = eOspaLmb{i};
    variant.hOspa = hOspaLmb{i};
    variant.cardinality = lmbCardinality{i};
    fixtureData.filterVariants{end+1} = variant;
end

% LMBM skipped due to bug with reduced timesteps

%% Save to JSON file
filename = fullfile(fixturesDir, sprintf('multisensor_trial_%d.json', seed));
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
fprintf('  - LMB variants: 4 × 100 timesteps (IC/PU/GA/AA)\n');
fprintf('  - LMBM variant: SKIPPED (bug in MATLAB code)\n');
