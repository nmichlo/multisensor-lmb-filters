classdef SimpleRng
    properties
        state  % uint64
    end
    methods
        function obj = SimpleRng(seed)
            obj.state = uint64(seed);
            if obj.state == 0
                obj.state = uint64(1);
            end
        end

        function [obj, val] = next_u64(obj)
            x = obj.state;
            x = bitxor(x, bitshift(x, 13));
            x = bitxor(x, bitshift(x, -7));
            x = bitxor(x, bitshift(x, 17));
            obj.state = x;
            val = x;
        end

        function [obj, val] = rand(obj, varargin)
            % Generate uniform random values in [0, 1)
            % [obj, val] = rand(obj) - single value
            % [obj, val] = rand(obj, n) - n x 1 vector
            % [obj, val] = rand(obj, rows, cols) - rows x cols matrix
            if nargin == 1
                % Scalar
                [obj, u] = obj.next_u64();
                val = double(u) / (2^64);
            elseif nargin == 2
                % Vector n x 1
                n = varargin{1};
                val = zeros(n, 1);
                for i = 1:n
                    [obj, u] = obj.next_u64();
                    val(i) = double(u) / (2^64);
                end
            else
                % Matrix rows x cols
                rows = varargin{1};
                cols = varargin{2};
                val = zeros(rows, cols);
                for i = 1:rows
                    for j = 1:cols
                        [obj, u] = obj.next_u64();
                        val(i, j) = double(u) / (2^64);
                    end
                end
            end
        end

        function [obj, val] = randn(obj, varargin)
            % Generate standard normal random values N(0,1)
            % [obj, val] = randn(obj) - single value
            % [obj, val] = randn(obj, n) - n x 1 vector
            % [obj, val] = randn(obj, rows, cols) - rows x cols matrix
            if nargin == 1
                % Scalar - Box-Muller transform
                [obj, u1] = obj.rand();
                [obj, u2] = obj.rand();
                val = sqrt(-2*log(u1)) * cos(2*pi*u2);
            elseif nargin == 2
                % Vector n x 1
                n = varargin{1};
                val = zeros(n, 1);
                for i = 1:n
                    [obj, u1] = obj.rand();
                    [obj, u2] = obj.rand();
                    val(i) = sqrt(-2*log(u1)) * cos(2*pi*u2);
                end
            else
                % Matrix rows x cols
                rows = varargin{1};
                cols = varargin{2};
                val = zeros(rows, cols);
                for i = 1:rows
                    for j = 1:cols
                        [obj, u1] = obj.rand();
                        [obj, u2] = obj.rand();
                        val(i, j) = sqrt(-2*log(u1)) * cos(2*pi*u2);
                    end
                end
            end
        end

        function [obj, val] = poissrnd(obj, lambda)
            % Knuth algorithm
            L = exp(-lambda);
            k = 0;
            p = 1.0;
            while true
                [obj, u] = obj.rand();
                p = p * u;
                if p <= L
                    break;
                end
                k = k + 1;
            end
            val = k;
        end
    end
end
