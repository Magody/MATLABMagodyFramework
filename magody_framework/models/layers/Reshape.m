%{
Made by: Danny Díaz
EPN - 2021
%}
classdef Reshape < Layer
    
    properties
        shape_input;
        shape_output = [];
    end
    
    methods
        
        function self = Reshape(shape_output, shape_input)
            % if empty constructor, then auto shape
            if nargin >= 1
                self.shape_output = shape_output;
            end
            if nargin >= 2
                self.init(shape_input);
            end
        end
        
        function shape_output = init(self, shape_input)
            self.shape_input = shape_input;
            
            if isempty(self.shape_output)
                self.shape_output = [prod(self.shape_input), []];
            end
            shape_output = self.shape_output;
        end
        
        
        
        function output = forward(self, input)
            m = size(input, 4);
            self.shape_input(4) = m;
            output = reshape(input, [self.shape_output, m]);
        end
        
        function input_gradient = backward(self, output_gradient, learning_rate) %#ok<INUSD>
            input_gradient = reshape(output_gradient, self.shape_input);
        end
        
    end
    
    
end