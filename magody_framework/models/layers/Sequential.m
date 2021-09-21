%{
Made by: Danny Díaz
EPN - 2021
%}
classdef Sequential < handle
    
    properties
        network;
        shape_input;
        shape_output;
    end
    methods
        
        function self = Sequential(network)
            
            self.network = network;
            
            if isempty(self.network)
                self.shape_input = 0;
                self.shape_output = 0;
            else
                len_network = length(network);
            
                first_layer = network{1};
                self.shape_input = first_layer.shape_input;

                self.shape_output = first_layer.shape_output;

                for index_layer=2:len_network
                    self.shape_output = network{index_layer}.init(self.shape_output);
                end
            end
            
            
            
    
        end
        
        function output = forward(self, x)
            len_network = length(self.network);
            output = x;
            for index_layer=1:len_network
                output = self.network{index_layer}.forward(output);
            end
        end
    
    end
    
    
end

