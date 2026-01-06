function run_benchmarks()
% RUN_BENCHMARKS Run LMB filter benchmarks from JSON scenarios
%
% Usage:
%   cd benchmarks && octave run_benchmarks.m
%   matlab -batch "cd benchmarks; run_benchmarks"
%
% Focus: Timing performance only. For accuracy evaluation, use separate tools.

    addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'common'));
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lmb'));

    scenario_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', ...
        'multisensor-lmb-filters-rs', 'benchmarks', 'scenarios');

    if ~exist(scenario_dir, 'dir')
        fprintf('Error: scenarios not found: %s\n', scenario_dir);
        return;
    end

    files = dir(fullfile(scenario_dir, '*.json'));
    if isempty(files)
        fprintf('Error: no scenarios in %s\n', scenario_dir);
        return;
    end

    TIMEOUT_SEC = 10;

    % Filter × Associator configs (MATLAB only supports single-sensor LMB)
    configs = {'LBP'; 'Gibbs'; 'Murty'};

    fprintf('%-22s | %-18s | %9s\n', 'Scenario', 'Filter', 'Time(ms)');
    fprintf('%s\n', repmat('-', 1, 55));

    for i = 1:length(files)
        fpath = fullfile(scenario_dir, files(i).name);
        scenario = load_json(fpath);
        name = strrep(files(i).name, '.json', '');

        % Preprocess measurements outside timing
        [model_base, measurements] = preprocess(scenario);

        for c = 1:length(configs)
            method = configs{c};
            filter_name = sprintf('LMB-%s', method);

            try
                t_ms = run_filter(model_base, measurements, method, TIMEOUT_SEC);
                fprintf('%-22s | %-18s | %9.1f\n', name, filter_name, t_ms);
            catch ME
                if ~isempty(strfind(ME.message, 'Timeout'))
                    fprintf('%-22s | %-18s | %9s\n', name, filter_name, 'TIMEOUT');
                else
                    fprintf('%-22s | %-18s | %9s\n', name, filter_name, 'ERROR');
                end
            end
        end
    end
end


function [model, measurements] = preprocess(scenario)
    % Build model and preprocess measurements (outside timing)
    model = build_model(scenario);

    n_steps = length(scenario.steps);
    measurements = cell(n_steps, 1);
    for t = 1:n_steps
        % Get first sensor's readings (single-sensor benchmark only)
        % JSON structure: sensor_readings[0] is a list of [x,y] measurements
        sr = scenario.steps(t).sensor_readings;

        % Handle various JSON parsing formats
        if iscell(sr)
            readings_cell = sr{1};  % First sensor
            if iscell(readings_cell)
                % Cell array of measurements - convert to matrix
                n = length(readings_cell);
                readings_mat = zeros(2, n);
                for i = 1:n
                    if iscell(readings_cell{i})
                        readings_mat(:, i) = [readings_cell{i}{1}; readings_cell{i}{2}];
                    else
                        readings_mat(:, i) = readings_cell{i}(:);
                    end
                end
                measurements{t} = readings_mat;
            elseif ismatrix(readings_cell) && ~isempty(readings_cell)
                % Already a matrix - transpose to 2 x N
                measurements{t} = readings_cell(:, 1:2)';
            else
                measurements{t} = [];
            end
        elseif ismatrix(sr) && ndims(sr) == 3
            % 3D array: sensors x measurements x coords
            measurements{t} = squeeze(sr(1, :, :))';  % First sensor, transpose to 2 x N
        else
            measurements{t} = [];
        end
    end
end


function t_ms = run_filter(model_base, measurements, method, timeout_sec)
    model = model_base;
    model.dataAssociationMethod = method;

    rng = SimpleRng(42);

    tic;
    runLmbFilter(rng, model, measurements);
    t_ms = toc * 1000;

    if t_ms > timeout_sec * 1000
        error('Timeout');
    end
end


function model = build_model(scenario)
    m = scenario.model;
    bounds = scenario.bounds;

    model.xDimension = 4;
    model.zDimension = 2;
    model.T = m.dt;

    % Motion model
    model.A = [eye(2), m.dt * eye(2); zeros(2), eye(2)];
    model.u = zeros(4, 1);

    q = m.process_noise_std^2;
    dt = m.dt;
    model.R = q * [dt^3/3*eye(2), dt^2/2*eye(2); dt^2/2*eye(2), dt*eye(2)];

    % Observation model
    model.C = [eye(2), zeros(2)];
    model.Q = m.measurement_noise_std^2 * eye(2);

    % Detection and clutter
    model.detectionProbability = m.detection_probability;
    model.survivalProbability = m.survival_probability;
    model.clutterRate = m.clutter_rate;

    obs_vol = (bounds(2) - bounds(1)) * (bounds(4) - bounds(3));
    model.observationSpaceLimits = [bounds(1), bounds(2); bounds(3), bounds(4)];
    model.observationSpaceVolume = obs_vol;
    model.clutterPerUnitVolume = m.clutter_rate / obs_vol;

    % Birth model
    birth_locs = m.birth_locations;
    model.numberOfBirthLocations = size(birth_locs, 1);
    model.birthLocationLabels = 1:model.numberOfBirthLocations;
    model.rB = 0.01 * ones(model.numberOfBirthLocations, 1);
    model.muB = cell(model.numberOfBirthLocations, 1);
    model.SigmaB = cell(model.numberOfBirthLocations, 1);
    for i = 1:model.numberOfBirthLocations
        model.muB{i} = birth_locs(i, :)';
        model.SigmaB{i} = 100 * eye(4);
    end

    % Object struct template
    object.birthLocation = 0;
    object.birthTime = 0;
    object.r = 0;
    object.numberOfGmComponents = 0;
    object.w = zeros(0, 1);
    object.mu = repmat({}, 0, 1);
    object.Sigma = repmat({}, 0, 1);
    object.trajectoryLength = 0;
    object.trajectory = [];
    object.timestamps = zeros(1, 0);
    model.object = repmat(object, 0, 1);

    % Birth parameters
    birthParams = repmat(object, model.numberOfBirthLocations, 1);
    for i = 1:model.numberOfBirthLocations
        birthParams(i).birthLocation = i;
        birthParams(i).birthTime = 0;
        birthParams(i).r = model.rB(i);
        birthParams(i).numberOfGmComponents = 1;
        birthParams(i).w = 1;
        birthParams(i).mu = model.muB(i);
        birthParams(i).Sigma = model.SigmaB(i);
        birthParams(i).trajectoryLength = 0;
        birthParams(i).trajectory = zeros(4, 100);
        birthParams(i).timestamps = zeros(1, 100);
    end
    model.birthParameters = birthParams;

    % Thresholds (matching Rust defaults)
    model.existenceThreshold = 1e-3;
    model.gmWeightThreshold = 1e-4;
    model.maximumNumberOfGmComponents = 100;
    model.minimumTrajectoryLength = 3;

    % Association parameters
    model.maximumNumberOfLbpIterations = 100;
    model.lbpConvergenceTolerance = 1e-6;
    model.numberOfSamples = 1000;
    model.numberOfAssignments = 25;
end


function scenario = load_json(fpath)
    fid = fopen(fpath, 'r');
    raw = fread(fid, inf, 'char');
    fclose(fid);
    json_str = char(raw');

    if exist('jsondecode', 'builtin') || exist('jsondecode', 'file')
        scenario = jsondecode(json_str);
    else
        error('jsondecode not available');
    end
end
