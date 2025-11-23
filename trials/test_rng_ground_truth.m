%% Test RNG sequence in ground truth generation
clc;
pkg load statistics;
setPath;

%% Generate model (seed 0)
modelRng = SimpleRng(0);
[modelRng, model] = generateModel(modelRng, 2.0, 0.95, 'LBP', 'Fixed');

%% Generate ground truth for FIRST 3 TIMESTEPS ONLY (seed 42)
rng = SimpleRng(42);

% Initialize
QChol = chol(model.Q, 'lower');
simulationLength = 3;  % Only first 3 timesteps
measurements = cell(simulationLength, 1);
for i = 1:simulationLength
    measurements{i} = cell(0, 1);
end

fprintf('=== MATLAB RNG Trace for Seed 42, First 3 Timesteps ===\n\n');

%% Clutter generation
fprintf('CLUTTER GENERATION:\n');
for i = 1:simulationLength
    fprintf('  t=%d:\n', i);
    [rng, numberOfClutterMeasurements] = rng.poissrnd(model.clutterRate);
    fprintf('    Poisson draw: %d clutter measurements\n', numberOfClutterMeasurements);
    measurements{i} = cell(numberOfClutterMeasurements, 1);
    for j = 1:numberOfClutterMeasurements
        [rng, randVec] = rng.rand(model.zDimension, 1);
        fprintf('      Clutter %d: rand()=[%.6f, %.6f]\n', j, randVec(1), randVec(2));
        measurements{i}{j} = model.observationSpaceLimits(:, 1) + 2 * model.observationSpaceLimits(:, 2) .* randVec;
        fprintf('                z=[%.4f, %.4f]\n', measurements{i}{j}(1), measurements{i}{j}(2));
    end
    fprintf('    Total clutter at t=%d: %d\n', i, length(measurements{i}));
end

%% Object trajectory and detection
fprintf('\nOBJECT DETECTION:\n');
numberOfObjects = numel(model.trajectory);
objectBirthTimes = [model.trajectory.birthTime];
objectDeathTimes = objectBirthTimes + [model.trajectory.trajectoryLength] - 1;
birthLocationIndex = [model.trajectory.birthLocation];
priorLocations = [model.trajectory.trajectory];

fprintf('Total objects: %d\n', numberOfObjects);

for i = 1:numberOfObjects
    trajectoryLength = objectDeathTimes(i) - objectBirthTimes(i) + 1;
    t = objectBirthTimes(i);
    x = priorLocations(:, i);

    fprintf('  Object %d (birth=t%d, death=t%d, length=%d):\n', i, t, objectDeathTimes(i), trajectoryLength);

    for j = 1:trajectoryLength
        if (j > 1)
            x = model.A * x + model.u;
            t = t + 1;
        end

        if t > simulationLength
            break;
        end

        % Detection check
        [rng, u] = rng.rand();
        fprintf('    t=%d j=%d: x=[%.2f, %.2f, %.2f, %.2f], detection rand()=%.6f', t, j, x(1), x(2), x(3), x(4), u);

        if (u < model.detectionProbability)
            fprintf(' -> DETECTED\n');
            [rng, noise] = rng.randn(1, model.zDimension);
            fprintf('             randn()=[%.6f, %.6f]\n', noise(1), noise(2));
            z = model.C * x + QChol * noise';
            fprintf('             z=[%.4f, %.4f]\n', z(1), z(2));
            measurements{t}{end+1} = z;
        else
            fprintf(' -> MISSED\n');
        end
    end
end

%% Summary
fprintf('\n=== SUMMARY ===\n');
for i = 1:simulationLength
    fprintf('t=%d: %d total measurements\n', i, length(measurements{i}));
end

fprintf('\n=== RNG State ===\n');
fprintf('Final RNG state: %lu\n', rng.state);
