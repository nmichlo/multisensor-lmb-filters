function [rng, groundTruth, measurements, groundTruthRfs] = generateGroundTruthShort(rng, model, varargin)
% GENERATEGROUNDTRUTHSHORT -- Short version with only 10 timesteps for quick testing
% This is a copy of generateGroundTruth.m with simulationLength = 10

%% Simple, hard-coded scenario
if (strcmp(model.scenarioType, 'Fixed'))
    numberOfObjects = 10;
    % Object birth times
    simulationLength = 10;  % CHANGED FROM 100 TO 10
    objectBirthTimes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1];  % All start at t=1
    objectDeathTimes = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10];  % All end at t=10
    % Object birth states
    birthLocationIndex = [1 2 3 4 1 4 1 2 3 4];
    priorLocations = [-80.0 -20.0 0.75 1.5;
        -20.0 80.0 -1.0 -2.0;
        0.0 0.0 -0.5 -1.0;
        40.0 -60.0 -0.25 -0.5;
        -80.0 -20.0 1.0 1.0;
        40.0 -60.0 -1.0 2.0;
        -80.0 -20.0 1.0 -0.5;
        -20.0 80.0 1.0 -1.0;
        0.0 0.0 1.0 -1.0;
        40.0 -60.0 -1.0 0.5]';
elseif (strcmp(model.scenarioType, 'Random'))
    if (nargin == 2)
        numberOfObjects = varargin{1};
    else
        error('You must specify the number of objects for a Random scenario');
    end
    simulationLength = 10;  % CHANGED FROM 100 TO 10
    objectBirthTimes = ones(1, numberOfObjects);
    objectDeathTimes = simulationLength * ones(1, numberOfObjects);
    objectDeathTimes(objectDeathTimes > simulationLength) = simulationLength;
    % Object birth states
    birthLocationIndex = randi(model.numberOfBirthLocations, 1, numberOfObjects);
    % Random positions
    priorLocations = zeros(4, numberOfObjects);
    for i = 1:numberOfObjects
        [rng, vals] = rng.rand(2, 1);
        priorLocations(1:2, i) = vals * 200 - 100;
        [rng, vals] = rng.randn(2, 1);
        priorLocations(3:4, i) = vals;
    end
end

%% Generate groundtruths
groundTruth = repmat({struct()}, 1, numberOfObjects);
for i = 1:numberOfObjects
    states = zeros(5, objectDeathTimes(i) - objectBirthTimes(i) + 1);
    states(:, 1) = [objectBirthTimes(i); priorLocations(:, birthLocationIndex(i))];
    for t = 2:(objectDeathTimes(i) - objectBirthTimes(i) + 1)
        states(:, t) = [states(1, t-1) + 1; model.A * states(2:end, t-1) + model.u];
    end
    groundTruth{i}.states = states;
    groundTruth{i}.birthTime = objectBirthTimes(i);
    groundTruth{i}.deathTime = objectDeathTimes(i);
    groundTruth{i}.birthLocationIndex = birthLocationIndex(i);
end

%% Generate measurements
measurements = repmat({}, simulationLength, 1);
groundTruthRfs.x = repmat({{}}, 1, simulationLength);
groundTruthRfs.mu = groundTruthRfs.x;
groundTruthRfs.Sigma = groundTruthRfs.x;
groundTruthRfs.cardinality = zeros(1, simulationLength);
for t = 1:simulationLength
    measurements{t} = {};
    % Clutter
    [rng, numberOfClutterMeasurements] = rng.poissrnd(model.clutterRate);
    for i = 1:numberOfClutterMeasurements
        [rng, randVec] = rng.rand(model.zDimension, 1);
        measurements{t}{end+1} = model.observationSpaceLimits(:, 1) + 2 * model.observationSpaceLimits(:, 2) .* randVec;
    end
    % Object-generated measurements
    for i = 1:numberOfObjects
        if (groundTruth{i}.birthTime <= t && t <= groundTruth{i}.deathTime)
            % Kalman filter
            indexInTrajectory = t - groundTruth{i}.birthTime + 1;
            x = groundTruth{i}.states(2:end, indexInTrajectory);
            mu = model.muB{birthLocationIndex(i)};
            Sigma = model.SigmaB{birthLocationIndex(i)};
            for tau = 1:(indexInTrajectory - 1)
                mu = model.A * mu + model.u;
                Sigma = model.A * Sigma * model.A' + model.R;
                [rng, noise] = rng.randn(1, model.zDimension);
                QChol = chol(model.Q, 'lower');
                z = model.C * x + QChol * noise';
                K = Sigma * model.C'/(model.C * Sigma * model.C' + model.Q);
                mu = mu + K *(z - model.C * mu);
                Sigma = (eye(model.xDimension) - K * model.C) * Sigma;
                x = model.A * x + model.u;
            end
            groundTruthRfs.x{t}{end+1} = x;
            groundTruthRfs.mu{t}{end+1} = mu;
            groundTruthRfs.Sigma{t}{end+1} = Sigma;
            groundTruthRfs.cardinality(t) = groundTruthRfs.cardinality(t) + 1;
            % Measurement
            [rng, isDetected] = rng.rand();
            if (isDetected < model.detectionProbability)
                [rng, measurementNoise] = rng.randn(model.zDimension, 1);
                measurements{t}{end+1} = model.C * x + model.Q^(1/2) * measurementNoise;
            end
        end
    end
end
end
