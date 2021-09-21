%{
Made by: Danny Díaz
EPN - 2021
%}
classdef ConvolutionalNeuralNetwork < handle
    
    properties
        sequential_conv_network;
        sequential_network;
        nnConfig;
        
        % alpha decay
        alpha;
    end
    methods
        
        function self = ConvolutionalNeuralNetwork(sequential_conv_network, sequential_network, nnConfig)
            self.sequential_conv_network = sequential_conv_network;
            self.sequential_network = sequential_network;
            self.nnConfig = nnConfig;
            self.alpha = self.nnConfig.learning_rate;
    
        end
        
        function history = train(self, X, Y, verbose_level)

            history = containers.Map();

            history_errors = zeros([1, self.nnConfig.epochs]);
            len_data_train = size(X, 4);
            num_batchs = ceil(len_data_train/self.nnConfig.batch_size);
            
            times_conv_network_forward = zeros([self.nnConfig.epochs, num_batchs]);
            times_network_forward = zeros([self.nnConfig.epochs, num_batchs]);
            times_conv_network_backward = zeros([self.nnConfig.epochs, num_batchs]);
            times_network_backward = zeros([self.nnConfig.epochs, num_batchs]);

            for e=1:self.nnConfig.epochs
                error = 0;
                index_batch = 1;
                for index_data=1:self.nnConfig.batch_size:len_data_train
        
                    batch_range = index_data:min(len_data_train, index_data+self.nnConfig.batch_size-1);

                    x = X(:, :, :, batch_range);
                    y = Y(batch_range, :);
                    
                    
                    t0 = tic;
                    features = self.sequential_conv_network.forward(x);
                    times_conv_network_forward(e, index_batch) = toc(t0);
                    t0 = tic;
                    output = self.sequential_network.forward(features);
                    times_network_forward(e, index_batch) = toc(t0);
                    

                    % error
                    error = error + sum(sum(self.nnConfig.functionLossCost(y', output), 1), 2)/self.nnConfig.batch_size;

                    if isnan(error) || sum(sum(isnan(output))) > 0
                        disp("Gradient exploding or vanishing");
                    end
                    
                    
                    [~, t_conv_network, t_network] = self.backward(y', output);

                    times_conv_network_backward(e, index_batch) = t_conv_network;
                    times_network_backward(e, index_batch) = t_network;
                    
                    index_batch = index_batch + 1;
                    
                end
                
                self.alpha = self.nnConfig.learning_rate/(1 + self.nnConfig.decay_rate_alpha * e);


                error = error / num_batchs;
                
                history_errors(1, e) = error;

                if mod(e, floor(self.nnConfig.epochs/10)) == 0 && verbose_level > 0
                    fprintf("%d/%d, error=%.4f\n", e, self.nnConfig.epochs, error);
                end
            end

            history('error') = history_errors;
            history('times_conv_network_forward') = times_conv_network_forward;
            history('times_network_forward') = times_network_forward;
            history('times_conv_network_backward') = times_conv_network_backward;
            history('times_network_backward') = times_network_backward;
                       
        end
        
        function [grad, time_conv_network_backward, time_network_backward] = backward(self, y, output)
            len_network = length(self.sequential_network.network);
            
            t0 = tic;
            grad = self.nnConfig.functionLossGradient(y, output);
            for index_layer=len_network:-1:1
                grad = self.sequential_network.network{index_layer}.backward(grad, self.alpha);
            end
            time_network_backward = toc(t0);
            
            len_conv_network = length(self.sequential_conv_network.network);
            t0 = tic;
            for index_layer=len_conv_network:-1:1
                grad = self.sequential_conv_network.network{index_layer}.backward(grad, self.alpha);
            end
            time_conv_network_backward = toc(t0);
            
            
        end
        
        function y_pred = predict(self, X)
            features = self.sequential_conv_network.forward(X);
            y_pred = self.sequential_network.forward(features);
        end
    
    end
    
    
end

