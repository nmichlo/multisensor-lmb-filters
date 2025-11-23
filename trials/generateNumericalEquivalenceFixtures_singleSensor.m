% GENERATENUMERICALEQUIVALENCEFIXTURES_SINGLESENSOR -- Phase 5.2 numerical equivalence fixtures
%
% This script generates comprehensive fixtures for exact numerical equivalence testing
% between MATLAB and Rust implementations.
%
% Coverage:
% - 5 filter variants: LMB-LBP, LMB-Gibbs, LMB-Murty's, LMBM-Gibbs, LMBM-Murty's
% - 5 seeds: 1, 42, 100, 1000, 12345
% - Full state estimates (not just OSPA metrics) for exact numerical comparison
%
% Output: fixtures/numerical_equivalence/single_sensor_seed{N}.json (5 files)

%% Admin
clc; close all;
pkg load statistics; % For Octave
setPath;

%% Configuration
seeds = [1, 42, 100, 1000, 12345];
lmbSimulationLength = 100;  % Full simulation for LMB
lmbmSimulationLength = 10;  % Reduced for LMBM (faster, validates core mechanics)

lmbDataAssociationMethods = {'LBP', 'Gibbs', 'Murty'};
lmbmDataAssociationMethods = {'Gibbs', 'Murty'};

%% Create fixtures directory
fixturesDir = fullfile(fileparts(mfilename('fullpath')), '..', 'fixtures', 'numerical_equivalence');
if ~exist(fixturesDir, 'dir')
    mkdir(fixturesDir);
    fprintf('Created directory: %s\n', fixturesDir);
end

%% Generate model (same for all trials, fixed seed 0)
fprintf('Generating model...\n');
modelRng = SimpleRng(0);
[modelRng, model] = generateModel(modelRng, 2, 0.95, 'LBP');

%% Process each seed
totalStartTime = tic;
for seedIdx = 1:numel(seeds)
    seed = seeds(seedIdx);
    seedStartTime = tic;
    fprintf('\n=== Processing seed %d (%d/%d) ===\n', seed, seedIdx, numel(seeds));

    % Generate ground truth for this seed
    fprintf('  [%s] Generating ground truth (full %d timesteps)...\n', datestr(now, 'HH:MM:SS'), lmbSimulationLength);
    trialRng = SimpleRng(seed);
    [trialRng, groundTruth, measurements, groundTruthRfs] = generateGroundTruth(trialRng, model);
    fprintf('  [%s] Ground truth generated (%d timesteps)\n', datestr(now, 'HH:MM:SS'), numel(measurements));

    % Initialize fixture structure
    fixtureData = struct();
    fixtureData.seed = seed;
    fixtureData.model = model;
    fixtureData.groundTruth = groundTruth;
    fixtureData.measurements = measurements;
    fixtureData.filterVariants = {};

    %% Run LMB filters (all methods, full simulation)
    for methodIdx = 1:numel(lmbDataAssociationMethods)
        method = lmbDataAssociationMethods{methodIdx};
        filterStartTime = tic;
        fprintf('  [%s] Running LMB-%s (%d timesteps)...', datestr(now, 'HH:MM:SS'), method, lmbSimulationLength);
        fflush(stdout);

        model.dataAssociationMethod = method;
        filterRng = SimpleRng(seed + 1000);
        [filterRng, stateEstimates] = runLmbFilter(filterRng, model, measurements);

        % Compute OSPA for validation
        [eOspa, hOspa, cardinality] = computeSimulationOspa(model, groundTruthRfs, stateEstimates);

        % Store complete results
        variant = struct();
        variant.name = ['LMB-' method];
        variant.filterType = 'LMB';
        variant.dataAssociationMethod = method;
        variant.simulationLength = lmbSimulationLength;
        variant.stateEstimates = stateEstimates;
        variant.eOspa = eOspa;
        variant.hOspa = hOspa;
        variant.cardinality = cardinality;

        fixtureData.filterVariants{end+1} = variant;
        fprintf(' done (%.1fs, final OSPA=%.4f)\n', toc(filterStartTime), eOspa(end));
    end

    %% Run LMBM filters (all methods, reduced simulation)
    % Create shortened measurements and ground truth for LMBM
    measurementsShort = measurements(1:lmbmSimulationLength);
    groundTruthRfsShort = struct();
    groundTruthRfsShort.x = groundTruthRfs.x(1:lmbmSimulationLength);
    groundTruthRfsShort.mu = groundTruthRfs.mu(1:lmbmSimulationLength);
    groundTruthRfsShort.Sigma = groundTruthRfs.Sigma(1:lmbmSimulationLength);

    for methodIdx = 1:numel(lmbmDataAssociationMethods)
        method = lmbmDataAssociationMethods{methodIdx};
        filterStartTime = tic;
        fprintf('  [%s] Running LMBM-%s (%d timesteps, REDUCED)...', datestr(now, 'HH:MM:SS'), method, lmbmSimulationLength);
        fflush(stdout);

        model.dataAssociationMethod = method;
        filterRng = SimpleRng(seed + 1000);
        [filterRng, stateEstimates] = runLmbmFilter(filterRng, model, measurementsShort);

        % Compute OSPA for validation
        [eOspa, hOspa, cardinality] = computeSimulationOspa(model, groundTruthRfsShort, stateEstimates);

        % Store complete results
        variant = struct();
        variant.name = ['LMBM-' method];
        variant.filterType = 'LMBM';
        variant.dataAssociationMethod = method;
        variant.simulationLength = lmbmSimulationLength;
        variant.stateEstimates = stateEstimates;
        variant.eOspa = eOspa;
        variant.hOspa = hOspa;
        variant.cardinality = cardinality;

        fixtureData.filterVariants{end+1} = variant;
        fprintf(' done (%.1fs, final OSPA=%.4f)\n', toc(filterStartTime), eOspa(end));
    end

    %% Save fixture to JSON
    filename = fullfile(fixturesDir, sprintf('single_sensor_seed%d.json', seed));
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

    seedElapsedTime = toc(seedStartTime);
    fprintf('  [%s] ✓ Seed %d complete (%.1fs, 5 variants)\n', datestr(now, 'HH:MM:SS'), seed, seedElapsedTime);
end

%% Summary
totalElapsedTime = toc(totalStartTime);
fprintf('\n=== COMPLETE ===\n');
fprintf('Total time: %.1fs\n', totalElapsedTime);
fprintf('Generated %d fixtures (5 seeds × 5 filter variants each)\n', numel(seeds));
fprintf('Output directory: %s\n', fixturesDir);
fprintf('\nFiles:\n');
for seedIdx = 1:numel(seeds)
    seed = seeds(seedIdx);
    filename = fullfile(fixturesDir, sprintf('single_sensor_seed%d.json', seed));
    fprintf('  - single_sensor_seed%d.json\n', seed);
end
