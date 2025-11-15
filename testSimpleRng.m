% Cross-language RNG equivalence test for MATLAB/Octave
% This script generates output that should match the Rust test exactly
%
% To verify equivalence with Rust, run:
%   octave testSimpleRng.m
% Then compare with Rust output:
%   cd /Users/nathanmichlo/Desktop/active/prak
%   cargo test --test test_rng_equivalence -- --nocapture

fprintf('Testing SimpleRng cross-language equivalence\n');
fprintf('Seed: 42\n\n');

% Test next_u64()
rng = SimpleRng(42);
fprintf('First 10 next_u64() values:\n');
for i = 1:10
    [rng, val] = rng.next_u64();
    fprintf('  %d: %s\n', i-1, num2str(val, '%.0f'));
end

% Test rand()
rng = SimpleRng(42);
fprintf('\nFirst 10 rand() values:\n');
for i = 1:10
    [rng, val] = rng.rand();
    fprintf('  %d: %.17e\n', i-1, val);
end

% Test randn()
rng = SimpleRng(42);
fprintf('\nFirst 10 randn() values:\n');
for i = 1:10
    [rng, val] = rng.randn();
    fprintf('  %d: %.17e\n', i-1, val);
end

% Test poissrnd()
rng = SimpleRng(42);
fprintf('\nFirst 10 poissrnd(5.0) values:\n');
for i = 1:10
    [rng, val] = rng.poissrnd(5.0);
    fprintf('  %d: %d\n', i-1, val);
end

% Test multiple seeds
seeds = [0, 1, 42, 12345, uint64(2^32-1), uint64(2^63-1)];
fprintf('\nTesting multiple seeds:\n');
for s = seeds
    rng = SimpleRng(s);
    [rng, val1] = rng.next_u64();
    [rng, val2] = rng.next_u64();
    [rng, val3] = rng.next_u64();
    fprintf('Seed %s: first 3 values = %s, %s, %s\n', ...
        num2str(s, '%.0f'), ...
        num2str(val1, '%.0f'), ...
        num2str(val2, '%.0f'), ...
        num2str(val3, '%.0f'));
end

% Test deterministic behavior
rng1 = SimpleRng(42);
rng2 = SimpleRng(42);
all_match = true;
for i = 1:10000
    [rng1, val1] = rng1.next_u64();
    [rng2, val2] = rng2.next_u64();
    if val1 ~= val2
        all_match = false;
        fprintf('ERROR: Mismatch at iteration %d\n', i);
        break;
    end
end
if all_match
    fprintf('\nDeterministic test: PASSED (10000 values matched)\n');
else
    fprintf('\nDeterministic test: FAILED\n');
end

% Test seed 0 handling
rng0 = SimpleRng(0);
rng1 = SimpleRng(1);
seed0_match = true;
for i = 1:100
    [rng0, val0] = rng0.next_u64();
    [rng1, val1] = rng1.next_u64();
    if val0 ~= val1
        seed0_match = false;
        fprintf('ERROR: Seed 0 handling mismatch at iteration %d\n', i);
        break;
    end
end
if seed0_match
    fprintf('Seed 0 handling test: PASSED (seed 0 == seed 1)\n');
else
    fprintf('Seed 0 handling test: FAILED\n');
end
