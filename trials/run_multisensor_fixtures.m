% Wrapper script to run all multisensor fixture generators
% Run from the multisensor-lmb-filters root directory

% Get the parent directory (root of multisensor-lmb-filters)
scriptDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(scriptDir);

% Set up paths using absolute paths
addpath(fullfile(rootDir, 'common'));
addpath(fullfile(rootDir, 'lmb'));
addpath(fullfile(rootDir, 'lmbm'));
addpath(fullfile(rootDir, 'marginalEvalulations'));
addpath(fullfile(rootDir, 'trials'));
addpath(fullfile(rootDir, 'multisensorLmb'));
addpath(fullfile(rootDir, 'multisensorLmbm'));

% Load statistics package
pkg load statistics;

% Run accuracy fixture generator
fprintf('\n=== Running Multisensor Accuracy Fixture Generator ===\n');
cd(fullfile(rootDir, 'trials'));
generateMultisensorAccuracyFixtures_quick;
cd(rootDir);

% Run clutter fixture generator
fprintf('\n=== Running Multisensor Clutter Fixture Generator ===\n');
cd(fullfile(rootDir, 'trials'));
generateMultisensorClutterFixtures_quick;
cd(rootDir);

% Run detection fixture generator
fprintf('\n=== Running Multisensor Detection Fixture Generator ===\n');
cd(fullfile(rootDir, 'trials'));
generateMultisensorDetectionFixtures_quick;
cd(rootDir);

fprintf('\n=== All fixtures generated successfully! ===\n');
