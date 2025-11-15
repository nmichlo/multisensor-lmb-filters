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

        function [obj, val] = rand(obj)
            [obj, u] = obj.next_u64();
            val = double(u) / (2^64);
        end

        function [obj, val] = randn(obj)
            % Box-Muller transform
            [obj, u1] = obj.rand();
            [obj, u2] = obj.rand();
            val = sqrt(-2*log(u1)) * cos(2*pi*u2);
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
