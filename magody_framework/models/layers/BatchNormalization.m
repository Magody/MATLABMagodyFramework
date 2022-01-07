%{
Made by: Danny Díaz
EPN - 2021
%}
classdef BatchNormalization < Layer
    
    properties
        name = "BatchNormalization";
        
        shape_input;
        shape_output;
        
        eps = 1e-7;
        
        % cache
        data_norm;
        data_centered;
        sample_std;
        
        % learnable   
        gamma = 1; % scale
        beta = 0; % shifting
        
        % running update and values
        momentum = 0.9;
        running_mean = 0;
        running_var = 0;
        
        % optimizer
        vdgamma = 0;
        vdbeta = 0;
        sdgamma = 0;
        sdbeta = 0;
    end
    
    methods
        
        function self = BatchNormalization(shape_input)
            if nargin == 1
               self.init(shape_input); 
            end
        end
        
        function shape_output = init(self, shape_input)
            self.shape_input = shape_input;
            self.shape_output = shape_input;
            shape_output = self.shape_output;
        end
        
        function output = forward(self, input, context)
            
            is_test = context("is_test");
            
            m = size(input, 2);
            
            if is_test
                % self.data_norm = input - self.running_mean ./ sqrt(self.running_var + self.eps);
                
                self.data_norm = input - mean(self.running_mean) ./ sqrt(mean(self.running_var) + self.eps);
                output = mean(self.gamma) .* self.data_norm + mean(self.beta);    
            else
                
                % the axis is vertical corresponding to each sample
                sample_mean = mean(input, 1);
                sample_var = var(input, 1);
                
                % running ponderate
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * sample_mean;
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * sample_var;
        
                
                
                self.sample_std = sqrt(sample_var + self.eps);
                self.data_centered = input - sample_mean;
                self.data_norm = self.data_centered ./ self.sample_std;
                
                % gamma scales, beta shifts
                output = self.gamma .* self.data_norm + self.beta;
            end
                
            
        end
        
        function input_gradient = backward(self, output_gradient, learning_rate) %#ok<INUSD>
            
            N = size(output_gradient, 1);
            
            dgamma = sum(output_gradient .* self.data_norm, 1);
            dbeta = sum(output_gradient, 1);
            
            dx_norm = output_gradient .* self.gamma;
            
            
            dx_centered = dx_norm ./ self.sample_std;
            dmean = -sum(dx_centered, 1) + 2/N * sum(self.data_centered, 1);
            dstd = sum(dx_norm .* self.data_centered .* -self.sample_std .^ -2, 1);
            dvar = dstd ./ 2 ./ self.sample_std;
            
            %{
            Pending to check:
              
            dx_norm = dout * gamma
            dx = 1/N / std * (N * dx_norm - 
                      dx_norm.sum(axis=0) - 
                      x_norm * (dx_norm * x_norm).sum(axis=0))    
            %}
            
            
            % ADAM tryhard
            b1 = 0.9;
            b2 = 0.999;
            
            % update momentum
            self.vdgamma = b1 * self.vdgamma + (1-b1) * dgamma;
            self.vdbeta = b1 * self.vdbeta + (1-b1) * dbeta;
            % update RMSprop
            self.sdgamma = b2 * self.sdgamma + (1-b2) * (dgamma .^ 2);
            self.sdbeta = b2 * self.sdbeta + (1-b2) * (dbeta .^ 2);
            
            % update learnable parameters
            self.gamma = self.gamma - learning_rate * (self.vdgamma./(sqrt(self.sdgamma) + self.eps));
            self.beta = self.beta - learning_rate * (self.vdbeta./(sqrt(self.sdbeta) + self.eps));
            
            
            input_gradient = dx_centered + (dmean + dvar .* 2 .* self.data_centered) / N;
            
        end
        
        
        
    end
    
    
end