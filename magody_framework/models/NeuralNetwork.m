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

            history = containers.Map();

            history_errors = zeros([1, self.nnConfig.epochs]);
            len_data_train = size(X, 1);
            num_batchs = ceil(len_data_train/self.nnConfig.batch_size);

            for e=1:self.nnConfig.epochs
                error = 0;
                for index_data=1:self.nnConfig.batch_size:len_data_train
                    batch_range = index_data:min(len_data_train, index_data+self.nnConfig.batch_size-1);

                    x = X(batch_range, :)';
                    y = Y(batch_range, :);

                    output = self.sequential_network.forward(x);
                    
                    % error
                    error = error + sum(sum(self.nnConfig.functionLossCost(y', output), 1), 2)/self.nnConfig.batch_size;

                    if isnan(error) || sum(sum(isnan(output))) > 0
                        disp("Gradient exploding or vanishing");
                    end
                    
                    
                    self.backward(y', output);
                    

                end
                
                self.alpha = self.nnConfig.learning_rate/(1 + self.nnConfig.decay_rate_alpha * e);

                error = error / num_batchs;
                
                history_errors(1, e) = error;

                if mod(e, floor(self.nnConfig.epochs/10)) == 0 && verbose_level > 0
                    fprintf("%d/%d, error=%.4f\n", e, self.nnConfig.epochs, error);
                end
            end

            history('error') = history_errors;
        end
        
        function grad = backward(self, y, output)
            len_network = length(self.sequential_network.network);
            grad = self.nnConfig.functionLossGradient(y, output);
            for index_layer=len_network:-1:1
                grad = self.sequential_network.network{index_layer}.backward(grad, self.alpha);
            end
        end
        
        function y_pred = predict(self, X)
            y_pred = self.sequential_network.forward(X');
        end
    
    end
    
    
end

