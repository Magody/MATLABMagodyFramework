%{
Made by: Danny Díaz
EPN - 2021
%}
classdef ConvolutionalNeuralNetwork < NeuralNetwork
    
    properties
        sequential_conv_network;
    end
    methods
        
        function self = ConvolutionalNeuralNetwork(sequential_conv_network, sequential_network, nnConfig)
            
            self = self@NeuralNetwork(sequential_network, nnConfig);
            
            self.sequential_conv_network = sequential_conv_network;
    
        end
        
    
    end
    
    methods (Access = public)
        % override methods
        
        function output = forwardFull(self, x, context)
            features = self.sequential_conv_network.forward(x, context);
            output = self.sequential_network.forward(features, context);
        end

        function grad = backward(self, y, output)
            len_network = length(self.sequential_network.network);

            grad = self.nnConfig.functionLossGradient(y, output);
            for index_layer=len_network:-1:1
                grad = self.sequential_network.network{index_layer}.backward(grad, self.alpha);
            end

            len_conv_network = length(self.sequential_conv_network.network);
            for index_layer=len_conv_network:-1:1
                grad = self.sequential_conv_network.network{index_layer}.backward(grad, self.alpha);
            end            

        end
    end
    
    
end

