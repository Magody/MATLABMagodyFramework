%{
Made by: Danny Díaz
EPN - 2021
%}
classdef QNeuralNetwork < handle
    
    properties
        
        % Neural network parameters
        shape_input;
        conv_network; % here is an optional network, {} for ignore it
        network;
        shape_output;
        
        nnConfig;
        qnnConfig;
        
        % epsilon decay
        epsilon;
        
        % alpha decay
        alpha;
        
        % aux
        use_convolutional;
    end
    methods
        
        function self = QNeuralNetwork(conv_network, network, nnConfig, qnnConfig)
            % conv_network is optional, if empty the forward is made only
            % with network
            self.conv_network = conv_network;
           
            self.network = network;
            self.nnConfig = nnConfig;
            self.qnnConfig = qnnConfig;
            
            self.epsilon = self.qnnConfig.initial_epsilon;
            self.alpha = self.nnConfig.learning_rate;
            
            self.use_convolutional = ~isempty(conv_network);
    
            if self.use_convolutional
                self.shape_input = conv_network{1}.shape_input;
            else
                self.shape_input = network{1}.shape_input;
            end
            
            self.shape_output = network{end}.shape_output;
        end
        
        function history = train(self, X, Y, episode, total_episodes, verbose_level) %#ok<INUSD>
            
            % is trained as multiclass classification
            
            size_X = size(X);
            input_dim = length(size_X);
            
            history = containers.Map();
            
            history_errors = zeros([1, self.nnConfig.epochs]);
            
            len_data_train = size(X, input_dim);
            
            
            num_batchs = ceil(len_data_train/self.nnConfig.batch_size);

            for epoch=1:self.nnConfig.epochs
                error = 0;
                for index_data=1:self.nnConfig.batch_size:len_data_train
        
                    batch_range = index_data:min(len_data_train, index_data+self.nnConfig.batch_size-1);

                    if input_dim == 2
                        x = X(:, batch_range);
                    elseif input_dim == 3
                        x = X(:, :, batch_range);
                    elseif input_dim == 4
                        x = X(:, :, :, batch_range);
                    end
                    y = Y(batch_range, :);
                    
                    output = self.forwardFull(x);
                    

                    % error
                    error = error + sum(sum(self.nnConfig.functionLossCost(y', output), 1), 2)/self.nnConfig.batch_size;

                    if isnan(error) || sum(sum(isnan(output))) > 0
                        disp("Error: gradient exploding or vanishing");
                    end
                    
                    self.backward(y', output);

                end
                
                
                error = error / num_batchs;
                
                history_errors(1, epoch) = error;
                
            end
            
            % epsilon decay
            self.epsilon = max(0.01, self.qnnConfig.initial_epsilon * log(exp(1) - ((exp(1) - 1) * (episode/total_episodes))));

            % alpha decay
            self.alpha = self.nnConfig.learning_rate/(1 + self.nnConfig.decay_rate_alpha * episode);
                

            history('history_errors') = history_errors;
        end
        
        function output = forwardFull(self, x)
            
            features = x;
            
            if self.use_convolutional
                features = self.forwardConv(x);
            end
            
            output = self.forward(features);
                    
        end
        
        
        function output = forwardConv(self, x)
            % x is in three dimensions: height, width, depth
            len_network = length(self.conv_network);
            output = x;
            for index_layer=1:len_network
                output = self.conv_network{index_layer}.forward(output);
            end
        end
        
        function output = forward(self, x)
            len_network = length(self.network);
            output = x;
            for index_layer=1:len_network
                output = self.network{index_layer}.forward(output);
            end
        end
        
        function grad = backward(self, y, output)
            len_network = length(self.network);
            grad = self.nnConfig.functionLossGradient(y, output);
            for index_layer=len_network:-1:1
                grad = self.network{index_layer}.backward(grad, self.alpha);
            end
            
            len_conv_network = length(self.conv_network);
            for index_layer=len_conv_network:-1:1
                grad = self.conv_network{index_layer}.backward(grad, self.alpha);
            end
        end
        
        function y_pred = predict(self, X)
            y_pred = self.forwardFull(X);
        end
        
        function [Qval, action_index] = selectAction(self, state, is_test)
            
            
            Qval = self.forwardFull(state)'; 
            [~, idx] = max(Qval); % obtengo indice de Qmax a partir de vector Qval

            if ~is_test
                % Epsilon-greedy action selection
                % Inicialmente hace solo exploracion, luego el valor de epsilon se va reduciendo a medida que tengo mas informacion
                % Si rand <= epsilon, obtengo un Q de manera aleatoria, el cual será diferente a Qmax (exploracion)

                v = rand;
                if v <= self.epsilon       
                    full_action_list = 1:self.network{end}.shape_output;   
                    actionList = full_action_list(full_action_list ~= idx);      %  Crea lista con las acciones q no tienen Qmax
                    idx_valid_action = randi([1 length(actionList)]);
                    idx = full_action_list(actionList(idx_valid_action));
                end
            end
            action_index = idx;  
            
        end
    
    end
    
    
end

