%{
Made by: Danny Díaz
EPN - 2021
%}
classdef NeuralNetwork < handle
    
    properties
        sequential_network;
        nnConfig;
        
        % alpha decay
        alpha;
    end
    methods
        
        function self = NeuralNetwork(sequential_network, nnConfig)
            self.sequential_network = sequential_network;
            self.nnConfig = nnConfig;
            self.alpha = self.nnConfig.learning_rate;
    
        end
        
        function history = train(self, X, Y, verbose_level)
            
            context = containers.Map({'is_test'}, {false});
            
            history = containers.Map();

            history_errors = zeros([1, self.nnConfig.epochs]);
            
            shape_input = size(X);
            num_dims = length(shape_input);
            
            is_two_dimensional = num_dims == 2;
            
            len_data_train = shape_input(end);
            num_batchs = ceil(len_data_train/self.nnConfig.batch_size);
                        
            for e=1:self.nnConfig.epochs
                error = 0;
                for index_data=1:self.nnConfig.batch_size:len_data_train
                    
                    batch_end = index_data+self.nnConfig.batch_size-1;
                    
                    if batch_end > len_data_train
                        batch_range = index_data:len_data_train;
                    else
                        batch_range = index_data:batch_end;
                    end
                    
                    % parametization for other integrations
                    if is_two_dimensional
                        x = X(:, batch_range);
                    else
                        x = X(:, :, :, batch_range);
                    end
                    y = Y(batch_range, :);
                    
                    output = self.forwardFull(x, context);
                    
                    % error
                    error = error + sum(sum(self.nnConfig.functionLossCost(y', output), 1), 2)/self.nnConfig.batch_size;
                    
                    self.backward(y', output);
                    

                end
                
                self.alpha = self.nnConfig.learning_rate/(1 + self.nnConfig.decay_rate_alpha * e);

                error = error / num_batchs;
                
                if isnan(error)
                    disp("Gradient exploding");
                end
                
                history_errors(1, e) = error;

                if mod(e, floor(self.nnConfig.epochs/10)) == 0 && verbose_level > 0
                    fprintf("%d/%d, error=%.4f\n", e, self.nnConfig.epochs, error);
                end
            end

            history('error') = history_errors;
        end
        
    end
    
    methods (Access = public)
        % this methods will be overriden by childs
        
        function output = forwardFull(self, x, context)
            output = self.sequential_network.forward(x, context);
        end
        
        function grad = backward(self, y, output)
            len_network = length(self.sequential_network.network);
            grad = self.nnConfig.functionLossGradient(y, output);
            for index_layer=len_network:-1:1
                grad = self.sequential_network.network{index_layer}.backward(grad, self.alpha);
            end
        end
        
        function y_pred = predict(self, X)
            context = containers.Map({'is_test'}, {true});
            y_pred = self.forwardFull(X, context);
        end
    
    end
    
    
end

