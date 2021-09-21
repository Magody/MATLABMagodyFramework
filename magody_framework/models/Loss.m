classdef Loss
    
    
    properties (Constant)
        eps = 1e-20;
    end
   
    methods (Static)
        
        function loss = mse(y_true, y_pred)
            loss = mean((y_true - y_pred) .^ 2);            
        end
        
        function loss = mse_derivative(y_true, y_pred)
            loss = 2 * (y_pred - y_true) / length(y_true);            
        end
        
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