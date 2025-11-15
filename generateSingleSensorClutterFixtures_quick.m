% Generate quick fixtures for single-sensor clutter trials with deterministic RNG
% Outputs JSON fixtures for Rust testing
% QUICK VERSION: Reduced timesteps and clutter rates for fast validation

clc; close all;
setPath;

% Trial configuration
seed = 42;
numberOfClutterReturns = [10 60];  % Only 2 rates for quick validation
numberOfExperimentsPerTrial = numel(numberOfClutterReturns);
detectionProbability = 0.95;
simulationLengthGibbs = 3;   % 3 timesteps for Gibbs variants (fast)
simulationLengthOther = 100;  % 100 timesteps for other variants

% Data association methods
lmbDataAssociationMethods = {'LBP', 'Gibbs', 'Murty'};
numberOfLmbAssociationMethods = numel(lmbDataAssociationMethods);
lmbmDataAssociationMethods = {'Gibbs', 'Murty'};
numberOfLmbmAssociationMethods = numel(lmbmDataAssociationMethods);

% Result storage
results = struct();
results.seed = seed;
results.clutterRates = numberOfClutterReturns;
results.detectionProbability = detectionProbability;
results.simulationLengthGibbs = simulationLengthGibbs;
results.simulationLengthOther = simulationLengthOther;
results.filterVariants = {};

fprintf('Generating single-sensor clutter sensitivity fixtures (seed=%d, quick mode)\n', seed);
fprintf('Simulation length: Gibbs=%d, Others=%d timesteps\n', simulationLengthGibbs, simulationLengthOther);
fprintf('Clutter rates: %s\n\n', mat2str(numberOfClutterReturns));

% Process each clutter rate
for i = 1:numberOfExperimentsPerTrial
    clutterRate = numberOfClutterReturns(i);
    fprintf('Clutter rate: %d returns\n', clutterRate);

    % Generate model with fixed seed for consistency across all methods
    rng = SimpleRng(0);
    [rng, model] = generateModel(rng, clutterRate, detectionProbability, 'LBP');

    % Generate FULL ground truth and measurements (100 timesteps)
    rng = SimpleRng(seed);
    [rng, groundTruthFull, measurementsFull, groundTruthRfsFull] = generateGroundTruth(rng, model);

    % LMB filters
    for j = 1:numberOfLmbAssociationMethods
        methodName = lmbDataAssociationMethods{j};
        fprintf('  LMB-%s... ', methodName);

        % Determine simulation length for this method
        if strcmp(methodName, 'Gibbs')
            simLen = simulationLengthGibbs;
            measurements = measurementsFull(1:simLen);
            groundTruthRfs = struct();
            groundTruthRfs.x = groundTruthRfsFull.x(1:simLen);
            groundTruthRfs.mu = groundTruthRfsFull.mu(1:simLen);
            groundTruthRfs.Sigma = groundTruthRfsFull.Sigma(1:simLen);
            groundTruthRfs.cardinality = groundTruthRfsFull.cardinality(1:simLen);
        else
            simLen = simulationLengthOther;
            measurements = measurementsFull;
            groundTruthRfs = groundTruthRfsFull;
        end

        % Set up model for this method
        model.dataAssociationMethod = methodName;

        % Run filter with seed+1000 (matching accuracy trials)
        rng = SimpleRng(seed + 1000);
        [rng, stateEstimates] = runLmbFilter(rng, model, measurements);

        % Compute OSPA metrics
        [eOspa, hOspa] = computeSimulationOspa(model, groundTruthRfs, stateEstimates);

        % Store mean values (aggregated across all timesteps)
        variantName = ['LMB-' methodName];
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

    % LMBM filters
    for j = 1:numberOfLmbmAssociationMethods
        methodName = lmbmDataAssociationMethods{j};
        fprintf('  LMBM-%s... ', methodName);

        % Determine simulation length for this method
        if strcmp(methodName, 'Gibbs')
            simLen = simulationLengthGibbs;
            measurements = measurementsFull(1:simLen);
            groundTruthRfs = struct();
            groundTruthRfs.x = groundTruthRfsFull.x(1:simLen);
            groundTruthRfs.mu = groundTruthRfsFull.mu(1:simLen);
            groundTruthRfs.Sigma = groundTruthRfsFull.Sigma(1:simLen);
            groundTruthRfs.cardinality = groundTruthRfsFull.cardinality(1:simLen);
        else
            simLen = simulationLengthOther;
            measurements = measurementsFull;
            groundTruthRfs = groundTruthRfsFull;
        end

        % Set up model for this method
        model.dataAssociationMethod = methodName;

        % Run filter with seed+1000
        rng = SimpleRng(seed + 1000);
        [rng, stateEstimates] = runLmbmFilter(rng, model, measurements);

        % Compute OSPA metrics
        [eOspa, hOspa] = computeSimulationOspa(model, groundTruthRfs, stateEstimates);

        % Store mean values
        variantName = ['LMBM-' methodName];
        if i == 1
            variantIdx = numel(results.filterVariants) + 1;
            results.filterVariants{variantIdx}.name = variantName;
            results.filterVariants{variantIdx}.eOspa = zeros(1, numberOfExperimentsPerTrial);
            results.filterVariants{variantIdx}.hOspa = zeros(1, numberOfExperimentsPerTrial);
        else
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
outputDir = fullfile(cd, 'fixtures', 'clutter');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

outputFile = fullfile(outputDir, sprintf('single_trial_%d_quick.json', seed));
jsonStr = jsonencode(results);
fid = fopen(outputFile, 'w');
fprintf(fid, '%s', jsonStr);
fclose(fid);

fprintf('Fixture saved to: %s\n', outputFile);
fprintf('Total variants: %d\n', numel(results.filterVariants));
fprintf('Clutter rates tested: %d\n', numberOfExperimentsPerTrial);
fprintf('File size: %.2f KB\n', dir(outputFile).bytes / 1024);
