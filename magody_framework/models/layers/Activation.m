%{
Made by: Danny Díaz
EPN - 2021
%}
classdef Activation < Layer
    
    properties
        input;
        activation;
        activation_derivative;
        
        shape_input;
        shape_output;
    end
    
    methods
        
        function self = Activation(activation_name)
            if activation_name == "sigmoid"
                self.activation = @Activation.sigmoid;
                self.activation_derivative = @Activation.sigmoid_derivative;
            elseif activation_name == "tanh"
                self.activation = @Activation.tan_h;
                self.activation_derivative = @Activation.tan_h_derivative;
            elseif activation_name == "relu"
                self.activation = @Activation.relu;
                self.activation_derivative = @Activation.relu_derivative;              
            elseif activation_name == "elu"
                self.activation = @Activation.elu;
                self.activation_derivative = @Activation.elu_derivative;              
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
        
        function input_gradient = backward(self, output_gradient, learning_rate) %#ok<INUSD>
            
            input_gradient = output_gradient .* self.activation_derivative(self.input);
            
        end
        
        
        
    end
    
    methods (Static)
        function y = sigmoid(x)
            y = 1 ./ (1 + exp(-x));
            
        end
        function y = sigmoid_derivative(x)
            s = Activation.sigmoid(x);
            y = s .* (1 - s);
        end


        function y = tan_h(x)
            y = tanh(x);
        end
        function y = tan_h_derivative(x)
            y = 1 - tanh(x) .^ 2;
        end
        
        function y = relu(x)
            y = max(0, x);
        end
        function y = relu_derivative(x)
            y = x > 0;
        end
        
        
        function y = softmax(x)
            % this is a final activation function
            ex = exp(x);
            y = ex./sum(ex);
        end
        
        function y = softmax_derivative(x)
            % the cross entropy is made outside with Class Loss
            y = x;
        end
        
        
        function y = elu(x)
            y = (x > 0) .* x + (x <= 0).*(exp(x) - 1);
        end
        function y = elu_derivative(x)
            threshold = 0;
            y = x > threshold;
        end
        
        
        
        
        
    end
    
    
end