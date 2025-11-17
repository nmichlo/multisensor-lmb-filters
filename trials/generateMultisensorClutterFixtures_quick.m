% Generate quick fixtures for multisensor clutter trials with deterministic RNG
% Outputs JSON fixtures for Rust testing
% QUICK VERSION: Reduced clutter rates for fast validation (no LMBM for speed)

clc; close all;
pkg load statistics; % For Octave
% NOTE: Run setPath before running this script

% Trial configuration
seed = 42;
numberOfClutterReturns = [10 60];  % Only 2 rates for quick validation
numberOfExperimentsPerTrial = numel(numberOfClutterReturns);
numberOfSensors = 3;
detectionProbabilities = [0.67 0.70 0.73];
q = [4 3 2];
simulationLength = 100;  % Full simulation for all variants

% Data association methods (exclude LMBM for speed)
lmbUpdateMethods = {'IC', 'PU', 'GA', 'AA'};
numberOfLmbUpdateMethods = numel(lmbUpdateMethods);

% Result storage
results = struct();
results.seed = seed;
results.clutterRates = numberOfClutterReturns;
results.detectionProbabilities = detectionProbabilities;
results.numberOfSensors = numberOfSensors;
results.simulationLength = simulationLength;
results.filterVariants = {};

fprintf('Generating multisensor clutter sensitivity fixtures (seed=%d, quick mode)\n', seed);
fprintf('Simulation length: %d timesteps\n', simulationLength);
fprintf('Clutter rates: %s\n', mat2str(numberOfClutterReturns));
fprintf('Sensors: %d\n\n', numberOfSensors);

% Process each clutter rate
for i = 1:numberOfExperimentsPerTrial
    clutterRate = numberOfClutterReturns(i);
    fprintf('Clutter rate: %d returns per sensor\n', clutterRate);

    % Generate model with fixed seed 0 (consistent across all methods)
    modelRng = SimpleRng(0);
    [modelRng, model] = generateMultisensorModel(modelRng, numberOfSensors, clutterRate * ones(1, numberOfSensors), detectionProbabilities, q, 'PU', 'LBP', 'Fixed');

    % Generate ground truth and measurements with trial seed
    trialRng = SimpleRng(seed);
    [trialRng, groundTruth, measurements, groundTruthRfs] = generateMultisensorGroundTruth(trialRng, model);

    % LMB filters
    for j = 1:numberOfLmbUpdateMethods
        methodName = lmbUpdateMethods{j};
        fprintf('  %s-LMB... ', methodName);

        % Run appropriate filter (no RNG parameter - filters don't use RNG currently)
        if strcmp(methodName, 'IC')
            stateEstimates = runIcLmbFilter(model, measurements);
        else
            model.lmbParallelUpdateMode = methodName;
            stateEstimates = runParallelUpdateLmbFilter(model, measurements);
        end

        % Compute OSPA metrics
        [eOspa, hOspa] = computeSimulationOspa(model, groundTruthRfs, stateEstimates);

        % Store mean values (aggregated across all timesteps)
        variantName = [methodName '-LMB'];
        if i == 1
            % Initialize arrays on first clutter rate
            variantIdx = numel(results.filterVariants) + 1;
            results.filterVariants{variantIdx}.name = variantName;
            results.filterVariants{variantIdx}.eOspa = zeros(1, numberOfExperimentsPerTrial);
            results.filterVariants{variantIdx}.hOspa = zeros(1, numberOfExperimentsPerTrial);
        else
            % Find existing variant
            variantIdx = 0;
            for k = 1:numel(results.filterVariants)
                if strcmp(results.filterVariants{k}.name, variantName)
                    variantIdx = k;
                    break;
                end
            end
        end

        results.filterVariants{variantIdx}.eOspa(i) = mean(eOspa);
        results.filterVariants{variantIdx}.hOspa(i) = mean(hOspa);

        fprintf('E-OSPA=%.4f, H-OSPA=%.4f\n', mean(eOspa), mean(hOspa));
    end

    fprintf('\n');
end

% Save to JSON
outputDir = fullfile(fileparts(mfilename('fullpath')), '..', 'fixtures', 'multisensor_clutter');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

outputFile = fullfile(outputDir, sprintf('multisensor_trial_%d_quick.json', seed));
jsonStr = jsonencode(results);
fid = fopen(outputFile, 'w');
fprintf(fid, '%s', jsonStr);
fclose(fid);

fprintf('Fixture saved to: %s\n', outputFile);
fprintf('Total variants: %d\n', numel(results.filterVariants));
fprintf('Clutter rates tested: %d\n', numberOfExperimentsPerTrial);
fprintf('File size: %.2f KB\n', dir(outputFile).bytes / 1024);
