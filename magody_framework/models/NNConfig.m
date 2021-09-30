%{
Made by: Danny Díaz
EPN - 2021
%}
classdef NNConfig < handle
    
    properties
    	% common parameters
        epochs = 100;
        learning_rate = 0.3;
        batch_size = 3;
        lambda = 0;
        
        % GD with momentum
        momentum_initial = 0.3;
        momentum_final = 0.9;
        num_epochs_to_increase_momentum = 50;
        
        % ADAM optimizer
        b1 = 0.9;
        b2 = 0.999;
        
        % loss and gradient associated functions
        functionLossCost;
        functionLossGradient;
        
        % alpha decay
        decay_rate_alpha = 0.9;
    end
    methods
        
        function self = NNConfig(epochs, learning_rate, batch_size, loss)
            
            
            self.epochs = epochs;
            self.learning_rate = learning_rate;
            self.batch_size = batch_size;
            
            if loss == "mse"
                self.functionLossCost = @Loss.mse;
                self.functionLossGradient = @Loss.mse_derivative;
            elseif loss == "softmax_cross_entropy"
                self.functionLossCost = @Loss.binary_cross_entropy;
                self.functionLossGradient = @Loss.softmaxGradient;
            elseif loss == "huber"
                % not implemented
                % self.functionLossCost = @Loss.huber;
                % self.functionLossGradient = @Loss.huber_derivative;
            end
    
        end
    
    end
    
    
end

