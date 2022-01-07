%{
Made by: Danny Díaz
EPN - 2021
%}
classdef ActivationOnlyForward < Layer
    
    properties
        input;
        activation;
        
        shape_input;
        shape_output;
        name = "ActivationOnlyForward";
    end
    
    methods
        
        function self = ActivationOnlyForward(activation_name)
            if activation_name == "softmax"
                self.activation = @Activation.softmax;             
            end
        end
        
        function shape_output = init(self, shape_input)
            self.shape_input = shape_input;
            self.shape_output = shape_input;
            shape_output = self.shape_output;
        end
        
        function output = forward(self, input, context)
            self.input = input;
            output = self.activation(self.input);
        end
        
        function input_gradient = backward(~, output_gradient, learning_rate) %#ok<INUSD>
            
            input_gradient = output_gradient;
            
        end
        
        
        
    end
    
    
end