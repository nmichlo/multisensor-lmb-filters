% GENERATESINGLE SENSORACCURACYFIXTURES -- Generate JSON fixtures for Rust testing
%
% This script generates deterministic fixtures for accuracy trial validation.
% Each fixture contains E-OSPA, H-OSPA, and cardinality data for all filter
% variants using a specific RNG seed.
%
% Fixture seeds: [1, 5, 10, 42, 50, 100, 500]
% Output: fixtures/accuracy/single_trial_{seed}.json

%% Admin
clc; close all;
pkg load statistics; % For Octave
setPath;

%% Configuration
% Seeds for which to generate fixtures
fixtureSeeds = [1, 5, 10, 42, 50, 100, 500];
numberOfFixtures = numel(fixtureSeeds);

% Simulation parameters (must match singleSensorAccuracyTrial.m)
simulationLength = 100;
lmbDataAssociationMethods = {'LBP', 'Gibbs', 'Murty'};
numberOfLmbAssociationMethods = numel(lmbDataAssociationMethods);
lmbmDataAssociationMethods = {'Gibbs', 'Murty'};
numberOfLmbmAssociationMethods = numel(lmbmDataAssociationMethods);

%% Generate model (same for all trials, fixed seed 0)
fprintf('Generating model...\n');
modelRng = SimpleRng(0);
[modelRng, model] = generateModel(modelRng, 2, 0.95, 'LBP'); % rBLbmbm = 0.06

%% Create fixtures directory
fixturesDir = fullfile(fileparts(mfilename('fullpath')), '..', 'fixtures', 'accuracy');
if ~exist(fixturesDir, 'dir')
    mkdir(fixturesDir);
    fprintf('Created directory: %s\n', fixturesDir);
end

%% Generate fixtures for each seed
startTime = tic;
for seedIdx = 1:numberOfFixtures
    seed = fixtureSeeds(seedIdx);
    seedStartTime = tic;
    fprintf('\n=== [%d/%d] Generating fixture for seed %d ===\n', seedIdx, numberOfFixtures, seed);

    % Initialize storage for this seed's results
    eOspaLmb = cell(1, numberOfLmbAssociationMethods);
    hOspaLmb = cell(1, numberOfLmbAssociationMethods);
    lmbCardinality = cell(1, numberOfLmbAssociationMethods);
    eOspaLmbm = cell(1, numberOfLmbmAssociationMethods);
    hOspaLmbm = cell(1, numberOfLmbmAssociationMethods);
    lmbmCardinality = cell(1, numberOfLmbmAssociationMethods);

    % Generate ground truth for this seed
    fprintf('  [%s] Generating ground truth...\n', datestr(now, 'HH:MM:SS'));
    trialRng = SimpleRng(seed);
    [trialRng, groundTruth, measurements, groundTruthRfs] = generateGroundTruth(trialRng, model);
    fprintf('  [%s] Ground truth generated (%.1fs)\n', datestr(now, 'HH:MM:SS'), toc(seedStartTime));

    %% Run LMB filters
    for i = 1:numberOfLmbAssociationMethods
        filterStartTime = tic;
        fprintf('  [%s] Running LMB-%s...', datestr(now, 'HH:MM:SS'), lmbDataAssociationMethods{i});
        model.dataAssociationMethod = lmbDataAssociationMethods{i};
        % Create filter RNG (use seed+1000 to differentiate from ground truth RNG)
        filterRng = SimpleRng(seed + 1000);
        [filterRng, stateEstimates] = runLmbFilter(filterRng, model, measurements);
        [eOspaLmb{i}, hOspaLmb{i}, lmbCardinality{i}] = computeSimulationOspa(model, groundTruthRfs, stateEstimates);
        fprintf(' done (%.1fs)\n', toc(filterStartTime));
    end

    %% Run LMBM filters
    for i = 1:numberOfLmbmAssociationMethods
        filterStartTime = tic;
        fprintf('  [%s] Running LMBM-%s (SLOW!)...', datestr(now, 'HH:MM:SS'), lmbmDataAssociationMethods{i});
        fflush(stdout);  % Force output flush for Octave
        model.dataAssociationMethod = lmbmDataAssociationMethods{i};
        % Create filter RNG (use seed+1000 to differentiate from ground truth RNG)
        filterRng = SimpleRng(seed + 1000);
        [filterRng, stateEstimates] = runLmbmFilter(filterRng, model, measurements);
        [eOspaLmbm{i}, hOspaLmbm{i}, lmbmCardinality{i}] = computeSimulationOspa(model, groundTruthRfs, stateEstimates);
        fprintf(' done (%.1fs)\n', toc(filterStartTime));
    end

    %% Build JSON structure
    fixtureData = struct();
    fixtureData.seed = seed;
    fixtureData.filterVariants = {};

    % Add LMB filter results
    for i = 1:numberOfLmbAssociationMethods
        variant = struct();
        variant.name = ['LMB-' lmbDataAssociationMethods{i}];
        variant.eOspa = eOspaLmb{i};
        variant.hOspa = hOspaLmb{i};
        variant.cardinality = lmbCardinality{i};
        fixtureData.filterVariants{end+1} = variant;
    end

    % Add LMBM filter results
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

    seedElapsedTime = toc(seedStartTime);
    fprintf('  [%s] ✓ Seed %d complete (%.1fs total)\n', datestr(now, 'HH:MM:SS'), seed, seedElapsedTime);

    % Estimate remaining time
    if seedIdx < numberOfFixtures
        avgTimePerSeed = toc(startTime) / seedIdx;
        remainingSeeds = numberOfFixtures - seedIdx;
        estimatedRemaining = avgTimePerSeed * remainingSeeds;
        fprintf('  Estimated time remaining: %.1f minutes (%.0f seeds left)\n', estimatedRemaining/60, remainingSeeds);
    end
end

totalElapsedTime = toc(startTime);
fprintf('\n=== All fixtures generated successfully ===\n');
fprintf('Total fixtures: %d\n', numberOfFixtures);
fprintf('Total time: %.1f minutes (%.0f seconds)\n', totalElapsedTime/60, totalElapsedTime);
fprintf('Output directory: %s\n', fixturesDir);
