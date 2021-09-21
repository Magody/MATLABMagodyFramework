%{
Made by: Danny Díaz
EPN - 2021
%}
classdef Convolutional < Layer
    
    properties
        
        shape_input;
        shape_output;
        
        
        input;
        output;
        
        num_filters;
        padding;
        stride;
        
        input_height;
        input_width;
        input_depth;
        shape_kernels;
        shape_kernel_matrix;
        kernels;
        biases;
        
        % ADAM optimizer
        vdw = 0;
        vdb = 0;
        sdw = 0;
        sdb = 0;
        t = 0;
    end
    
    methods
        
        function self = Convolutional(...
                shape_kernel_matrix, num_filters, padding, stride, shape_input ...
        )
            
            self.shape_kernel_matrix = shape_kernel_matrix;
            self.num_filters = num_filters;
            self.padding = padding;
            self.stride = stride;
            
            if nargin == 5
                self.init(shape_input);
            end
            
            
        end
        
        function shape_output = init(self, shape_input)
            self.shape_input = shape_input;
            self.input_height = shape_input(1);
            self.input_width = shape_input(2);
            self.input_depth = shape_input(3);
            
            
            self.shape_output = getOutputShape(self.input_height, self.input_width, self.shape_kernel_matrix, self.padding, self.stride, self.num_filters);
            self.shape_kernels = [self.shape_kernel_matrix, self.input_depth, self.num_filters];
            
            
            previous_neurons = 1;
            for i=1:length(self.shape_input)
                previous_neurons = previous_neurons * self.shape_input(i);
            end
            
            self.kernels = getCNNWeights(0, 0.5, self.shape_kernels, previous_neurons);
            self.biases = getCNNWeights(0, 0.5, self.shape_output, previous_neurons);
            
            shape_output = self.shape_output;
        end
        
        function output = forwardNoVectorized(self, input)
            self.t = self.t + 1;
            
            m = size(input, 4);
            
            self.input = input;
            output = zeros([self.shape_output, m]);
            
            correlation_shape = size(output, 1:2);
            
            for sample=1:m
                % copy of bias
                self.output = reshape(self.biases(:), self.shape_output); % zeros(self.shape_output); reshape(self.biases(:), self.shape_output);
                
                for i=1:self.num_filters
                    for j=1:self.input_depth
                        self.output(:, :, i) = self.output(:, :, i) + crossCorrelation2D(self.input(:, :, j, sample), self.kernels(:, :, j, i), "valid", correlation_shape);
                    end
                end
                output(:, :, :, sample) = self.output;
            end
            
            
            
        end
        
        function output = forward(self, input)
            self.t = self.t + 1;
            
            channels = size(input, 3);
            m = size(input, 4);
            
            self.input = input;
            
            
            [vectorized_images, shape_new_image] = vectorizeImages(input, self.shape_kernel_matrix);
            vectorized_kernels = vectorizeKernels(self.kernels);
            % simple crossCorrelation2D, no rot180
            % each vectorized for each sample
            output = zeros([self.shape_output, m]);
            for sample=1:m
                correlation = transpose(vectorized_images(:, :, sample) * vectorized_kernels);
                correlation_reshaped = reshapeCorrelation(correlation, channels, shape_new_image) + self.biases;
                output(:, :, :, sample) = correlation_reshaped;
            end
            
            
        end
        
        function input_gradient = backward(self, output_gradient, learning_rate)
            
            kernels_gradient = zeros(self.shape_kernels);
            input_gradient = zeros(self.shape_input);
            m = size(output_gradient, 4);
            
            correlation_shape_kernels_gradient = size(kernels_gradient, 1:2);
            correlation_shape_input_gradient = size(input_gradient, 1:2);

            for sample=1:m
                for i=1:self.num_filters
                    for j=1:self.input_depth
                        kernels_gradient(:, :, j, i) = kernels_gradient(:, :, j, i) + crossCorrelation2D(self.input(:, :, j, sample), output_gradient(:, :, i, sample), "valid", correlation_shape_kernels_gradient);
                        kernel180 = rot90(self.kernels(:, :, j, i), 2);
                        input_gradient(:, :, j) = input_gradient(:, :, j) + crossCorrelation2D(output_gradient(:, :, i, sample), kernel180, "full", correlation_shape_input_gradient);
                    end
                end
            end
            
            
            % Adam
            b1 = 0.9;
            b2 = 0.999;
            eps = 1e-8;
            
            bias_gradient = sum(output_gradient, 4);
            
            % update momentum
            self.vdw = b1 * self.vdw + (1-b1) * kernels_gradient;
            self.vdb = b1 * self.vdb + (1-b1) * bias_gradient;
            % update RMSprop
            self.sdw = b2 * self.sdw + (1-b2) * (kernels_gradient .^ 2);
            self.sdb = b2 * self.sdb + (1-b2) * (bias_gradient .^ 2);
            
            
            % bias correction
            %{
            self.vdw = self.vdw ./ (1 - (b1 ^ t));
            self.vdb = self.vdb ./ (1 - (b1 ^ t));
            self.sdw = self.sdw ./ (1 - (b2 ^ t));
            self.sdb = self.sdb ./ (1 - (b2 ^ t));
            %}
            
            
            self.kernels = self.kernels - learning_rate * (self.vdw./(sqrt(self.sdw) + eps));
            self.biases = self.biases - learning_rate * (self.vdb./(sqrt(self.sdb) + eps));
            %{
            
            self.kernels = self.kernels - learning_rate * kernels_gradient;
            self.biases = self.biases - learning_rate * sum(output_gradient, 4);
            
            self.kernels = self.kernels - learning_rate * (self.vdw./(sqrt(self.sdw) + eps));
            self.biases = self.biases - learning_rate * (self.vdb./(sqrt(self.sdb) + eps));
            
            
            %}
            
            
            
        end
        
    end
    
    
end