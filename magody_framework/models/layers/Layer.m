%{
Made by: Danny Díaz
EPN - 2021
%}
classdef (Abstract) Layer < handle
    
    properties (Abstract)
        name;
        shape_input;
        shape_output; 
    end
    
    methods (Abstract)
        % return output
        output = forward(self, input, context);
        % update parameters and return input gradient
        input_gradient = backward(self, output_gradient, learning_rate);
        % init shapes and others
        shape_output = init(self, shape_input);
    end
    
    
end

