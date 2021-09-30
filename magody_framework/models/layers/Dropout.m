%{
Made by: Danny Díaz
EPN - 2021
%}
classdef Dropout < Layer
    
    properties
        
        shape_input;
        shape_output;
        
        dropout_rate;
        mask;
    end
    
    methods
        
        function self = Dropout(dropout_rate, shape_input)
            self.dropout_rate = dropout_rate;
            
            if nargin == 2
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
                output = input;
            else
                keep_probability = 1 - self.dropout_rate;
                % get random from uniform, vectorized operation
                
                % self.mask = binornd(1, keep_probability * ones(self.shape_input));
                self.mask = rand(self.shape_input) < keep_probability;
                
                scale = 0;
                if keep_probability > 0
                   scale = 1/keep_probability; 
                end
                
                % arrange the mask to m examples in vectorization
                self.mask = repmat(self.mask, [1, m]);
                output = self.mask .* input * scale;
                
            end
        end
        
        function input_gradient = backward(self, output_gradient, learning_rate) %#ok<INUSD>
            
            input_gradient = output_gradient .* self.mask;
            
        end
        
        
        
    end
    
    
end