classdef Loss
    
    
    properties (Constant)
        eps = 1e-20;
        huber_k = 1;
    end
   
    methods (Static)
        
        % mse
        function loss = mse(y_true, y_pred)
            loss = mean((y_true - y_pred) .^ 2);            
        end
        
        function loss = mse_derivative(y_true, y_pred)
            loss = 2 * (y_pred - y_true) / length(y_true);            
        end
        
        % huber
        function loss = huber(y_true, y_pred)
            delta = Loss.huber_k;
            if nargin == 3
                delta = k;
            end
            
            residual = y_true - y_pred;
            residual_abs = abs(residual);
            
            value_residual_less_equal = 0.5 * residual .^ 2;
            value_residual_greater = delta * residual_abs + 0.5 * delta * delta;
            
            less_equal_than_delta = residual_abs <= delta;
            
            loss_result = zeros(size(less_equal_than_delta));
            
            for i=1:length(less_equal_than_delta)
                if less_equal_than_delta(i)
                    loss_result(i) = value_residual_less_equal(i);
                else
                    loss_result(i) = value_residual_greater(i);
                end
            end
            
            loss = mean(loss_result);
        end
        
        function loss = huber_derivative(y_true, y_pred)
            
            loss = sign((y_true - y_pred))/2;
            %{
            delta = Loss.huber_k;
            if nargin == 3
                delta = k;
            end
            
            residual = y_true - y_pred;
            
            loss = zeros(size(residual));
            
            % clip function, also can be done with signum function
            for i=1:length(residual)
                r = residual(i);
                if r < -delta
                    loss(i) = -delta;
                elseif -delta <= r && r <= delta
                    loss(i) = r;
                elseif r > delta
                    loss(i) = delta;
                end
            end
            %}
                 
        end
        
        % log cosh loss
        function loss = logcosh(y_true, y_pred)
            loss = sum(log(cosh(y_true - y_pred)));
        end
        
        function loss = logcosh_derivative(y_true, y_pred)
            loss = tanh(y_true - y_pred);
        end
        
        
        % cross entropy
        function loss = binary_cross_entropy(y_true, y_pred)
            % eps for preventing log(0)
            loss = mean(-y_true .* log(y_pred+Loss.eps) - (1 - y_true) .* log(1 - y_pred + Loss.eps));            
        end
        
        function loss = binary_cross_entropy_derivative(y_true, y_pred)
            loss = ((1 - y_true) ./ (1 - y_pred) - y_true ./ y_pred) / length(y_true);
        end
        
        function loss = softmaxGradient(y_true, y_pred)
            loss = y_pred - y_true;            
        end
        
    
    end
    
end
