%% Check if MATLAB rounds r values
clc;

% Test case: two values that sum to almost 1.0
tau1 = 0.43795963770742691;
tau2 = 0.56204036229257304;

r_sum = tau1 + tau2;

fprintf('tau1 = %.17g\n', tau1);
fprintf('tau2 = %.17g\n', tau2);
fprintf('r_sum = %.17g\n', r_sum);
fprintf('r_sum == 1.0: %d\n', r_sum == 1.0);
fprintf('r_sum - 1.0 = %.17g\n', r_sum - 1.0);
fprintf('Bit representation of r_sum: %016x\n', typecast(r_sum, 'uint64'));
fprintf('Bit representation of 1.0:   %016x\n', typecast(1.0, 'uint64'));

% Check if MATLAB's sum() function has special behavior
vec = [tau1, tau2];
r_matlab_sum = sum(vec);
fprintf('\nUsing sum() function:\n');
fprintf('sum([tau1, tau2]) = %.17g\n', r_matlab_sum);
fprintf('sum() == 1.0: %d\n', r_matlab_sum == 1.0);
fprintf('Bit representation: %016x\n', typecast(r_matlab_sum, 'uint64'));
