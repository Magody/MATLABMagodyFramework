%{
Made by: Danny Díaz
EPN - 2021
%}
classdef Convolutional < Layer
    
    properties
        
        % init weights
        init_mean = 0;
        init_std = 1;
        
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
        bias;
        
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
            
            
            if nargin >= 5
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
            
            
            % previous_neurons = prod(self.shape_kernel_matrix)*self.input_depth;
            % prod(self.shape_input);
            % 
            
            % kaiming he https://www.telesens.co/2018/04/09/initializing-weights-for-the-convolutional-and-fully-connected-layers/
            
            n = self.input_depth * prod(self.shape_kernel_matrix);
            
            % sigma = sqrt(2/n);
            % normrnd(0, sigma, self.shape_kernels);
            
            
            self.kernels = getWeights(self.init_mean, self.init_std, self.shape_kernels, n, "He");
            self.bias = zeros(self.shape_output);
            
            shape_output = self.shape_output;
        end
        
        
        
        
        
        function output = forward(self, input, context)
            % SylvainGugger shape https://sgugger.github.io/convolution-in-depth.html
            
            self.t = self.t + 1;
            
            
            % channels = size(input, 3);
            [n, p, ch, mb] = size(input);
            
            self.input = arr2vec(input, self.shape_kernels(1:2));
            
            bias_reshape = reshape(self.bias(:), [prod(self.shape_output(1:2)), self.shape_output(3)]);
            output = zeros([self.shape_output, mb]);
            
            kernels_reshape = reshape(self.kernels, [prod(self.shape_kernels(1:3)), self.shape_kernels(4)]);
            for sample=1:mb
                output(:, :, :, sample) = reshape(bias_reshape + self.input(:, :, sample) * kernels_reshape, [self.shape_output]);
            end
        end
        
        function input_gradient = backward(self, output_gradient, learning_rate)
            
            kernels_gradient = zeros(self.shape_kernels);
            
            [n1, p1, ch_out, mb] = size(output_gradient);
            grad = reshape(output_gradient, [n1*p1, ch_out, mb]);
            
            for sample=1:mb
                weight = grad(:, :, sample)' * self.input(:, :, sample);
                kernels_gradient(:, :, :, :) = kernels_gradient(:, :, :, :) + reshape(weight', self.shape_kernels);
            end
            
            kernels_gradient = kernels_gradient ./ mb;
            
            kernel = reshape(self.kernels, [prod(self.shape_kernels(1:3)), self.shape_kernels(4)]);
            
            kg = zeros([size(grad, 1), size(kernel, 1), mb]);
            
            for sample=1:mb
                kg(:, :, sample) = grad(:, :, sample) * kernel';
            end
            
            input_gradient = vec2arr(kg, self.shape_kernels(1:2), self.shape_input(1:2));
            
            
            
            % Adam
            b1 = 0.9;
            b2 = 0.999;
            eps = 1e-8;
            
            bias_gradient = mean(output_gradient, 4);
            
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
            self.bias = self.bias - learning_rate * (self.vdb./(sqrt(self.sdb) + eps));
            
            
            
            
        end
        
        
        
        
        function output = forwardSJLiXu(self, input, context)
            
            % Jimmy SJ. Ren Li Xu, On Vectorization of Deep Convolutional Neural Networks for Vision Tasks
            self.t = self.t + 1;
            
            % channels = size(input, 3);
            m = size(input, 4);
            
            self.input = input;
            
            % t1 = tic;
            [vectorized_images, shape_new_image] = vectorizeImages(input, self.shape_kernel_matrix);
            vectorized_kernels = vectorizeKernels(self.kernels);
            
            
            % t2 = toc(t1);
            
            % fprintf("Seg1: %.4f\n", t2);
           
            % t3 = tic;
            
            % filters X samples, each sample can be reshaped to correlation
            correlation_rolled = transpose(vectorized_kernels * vectorized_images);
            elements_of_correlation = prod(shape_new_image);
            correlation_blocks = zeros([elements_of_correlation, m * self.num_filters]);
            
            bias_reshape = reshape(self.bias, [prod(self.shape_output(1:2)), self.shape_output(3)]);
            for sample=1:m
                
                index_begin_image = (sample-1) * elements_of_correlation + 1;
                index_end_image = sample * elements_of_correlation;
                
                index_begin_filter = (sample-1) * self.num_filters + 1;
                index_end_filter = sample * self.num_filters;
                
                correlation_blocks(1:elements_of_correlation, index_begin_filter:index_end_filter) = bias_reshape + correlation_rolled(index_begin_image:index_end_image, 1:self.num_filters);
            end
            
            output = reshape(correlation_blocks, [self.shape_output, m]);
            
            
            % t4 = toc(t3);
            
            % fprintf("Seg2, per sample: %.4f of %d samples\n", t4/m, m);
            
            
        end
        
        function output = forwardNoVectorized(self, input)
            self.t = self.t + 1;
            
            m = size(input, 4);
            
            self.input = input;
            output = zeros([self.shape_output, m]);
            
            correlation_shape = size(output, 1:2);
            
            for sample=1:m
                % copy of bias, here we sum bias
                self.output = reshape(self.bias(:), self.shape_output); % zeros(self.shape_output); reshape(self.bias(:), self.shape_output);
                
                for i=1:self.num_filters
                    for j=1:self.input_depth
                        self.output(:, :, i) = self.output(:, :, i) + crossCorrelation2D(self.input(:, :, j, sample), self.kernels(:, :, j, i), "valid", correlation_shape);
                    end
                end
                output(:, :, :, sample) = self.output;
            end
            
            
            
        end
        
        
        function input_gradient = backwardNoVectorized(self, output_gradient, learning_rate)
            
            kernels_gradient = zeros(self.shape_kernels);
            m = size(output_gradient, 4);
            input_gradient = zeros([self.shape_input, m]);
            
            correlation_shape_kernels_gradient = size(kernels_gradient, 1:2);
            correlation_shape_input_gradient = size(input_gradient, 1:2);

            for sample=1:m
                for i=1:self.num_filters
                    for j=1:self.input_depth
                        try
                            kernels_gradient(:, :, j, i) = kernels_gradient(:, :, j, i) + crossCorrelation2D(self.input(:, :, j, sample), output_gradient(:, :, i, sample), "valid", correlation_shape_kernels_gradient);
                        catch ME
                            disp(ME);
                        end
                        
                        kernel180 = rot90(self.kernels(:, :, j, i), 2);
                        input_gradient(:, :, j, sample) = input_gradient(:, :, j, sample) + crossCorrelation2D(output_gradient(:, :, i, sample), kernel180, "full", correlation_shape_input_gradient);
                    end
                end
            end
            
            
            % Adam
            b1 = 0.9;
            b2 = 0.999;
            eps = 1e-8;
            
            bias_gradient = mean(output_gradient, 4);
            
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
            self.bias = self.bias - learning_rate * (self.vdb./(sqrt(self.sdb) + eps));
            %{
            
            self.kernels = self.kernels - learning_rate * kernels_gradient;
            self.bias = self.bias - learning_rate * mean(output_gradient, 4);
            
            self.kernels = self.kernels - learning_rate * (self.vdw./(sqrt(self.sdw) + eps));
            self.bias = self.bias - learning_rate * (self.vdb./(sqrt(self.sdb) + eps));
            
            
            %}
            
            
            
        end
        
    end
    
    
end