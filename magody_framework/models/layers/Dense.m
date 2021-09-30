%{
Made by: Danny Díaz
EPN - 2021
%}
classdef Dense < Layer
    
    properties
        input;
        weights;
        bias;
        
        
        shape_input;
        shape_output;
        
        % ADAM optimizer
        vdw = 0;
        vdb = 0;
        sdw = 0;
        sdb = 0;
        t = 0;
        
        % init mode
        mode;
    end
    
    methods
        
        function self = Dense(neurons_output, mode, neurons_input)
            % shape_input and shape_output are scalars
            self.shape_output = [neurons_output, 1];
            
            if nargin >= 2
                self.mode = mode;
            else
                self.mode = "xavier";
            end
            
            if nargin >= 3
                self.init([neurons_input, 1]);
            end
            
            
            
            
            
        end
        
        function shape_output = init(self, shape_input)
            self.shape_input = shape_input;
            shape_output = self.shape_output;
            self.weights = getWeights(0, 0.5, [shape_output(1), shape_input(1)], self.mode);
            self.bias = getWeights(0, 0.5, [shape_output(1), 1], self.mode);
        end
        
        function output = forward(self, input, context)
            % input = X => each col is an example
            self.input = input;
            output = self.weights * self.input + self.bias;
            
            self.t = self.t + 1;
            
        end
        
        function input_gradient = backward(self, output_gradient, learning_rate)
            
            weights_gradient = output_gradient * transpose(self.input);
            bias_gradient = mean(output_gradient, 2);
            
            input_gradient = transpose(self.weights) * output_gradient;
            
            
            % Adam
            b1 = 0.9;
            b2 = 0.999;
            eps = 1e-8;
            
            % update momentum
            self.vdw = b1 * self.vdw + (1-b1) * weights_gradient;
            self.vdb = b1 * self.vdb + (1-b1) * bias_gradient;
            % update RMSprop
            self.sdw = b2 * self.sdw + (1-b2) * (weights_gradient .^ 2);
            self.sdb = b2 * self.sdb + (1-b2) * (bias_gradient .^ 2);
            
            
            % bias correction
            %{
            self.vdw = self.vdw ./ (1 - (b1 ^ t));
            self.vdb = self.vdb ./ (1 - (b1 ^ t));
            self.sdw = self.sdw ./ (1 - (b2 ^ t));
            self.sdb = self.sdb ./ (1 - (b2 ^ t));
            %}
            
            self.weights = self.weights - learning_rate * (self.vdw./(sqrt(self.sdw) + eps));
            self.bias = self.bias - learning_rate * (self.vdb./(sqrt(self.sdb) + eps));
            %{
            self.weights = self.weights - learning_rate * weights_gradient;
            self.bias = self.bias - learning_rate * bias_gradient;
            
            self.weights = self.weights - learning_rate * (self.vdw./(sqrt(self.sdw) + eps));
            self.bias = self.bias - learning_rate * (self.vdb./(sqrt(self.sdb) + eps));
            
            
            %}
        end        
        
    end
    
    
end