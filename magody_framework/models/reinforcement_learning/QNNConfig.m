%{
Made by: Danny Díaz
EPN - 2021
%}
classdef QNNConfig < handle
    
    properties
        
        gamma = 1;
        initial_epsilon;
    end
    methods
        
        function self = QNNConfig(gamma, epsilon)
            self.gamma = gamma;
            self.initial_epsilon = epsilon;
        end
    
    end
    
    
end

