
function [vectorized, shape_new_image] = vectorizeImages(images, kernel_size)
    % images is 4-D matrix, [height, width, channels, samples]

    
    [rows, cols, channels, samples] = size(images);

    limit_rows = rows-kernel_size(1)+1;
    limit_cols = cols-kernel_size(2)+1;
    
    
    shape_new_image = [limit_rows, limit_cols];


    elements_per_channel = kernel_size(1) * kernel_size(2);
    elements_per_sample = elements_per_channel * channels;
    vectorized = zeros([limit_rows * limit_cols, elements_per_sample, samples]);



    for sample=1:samples

        % index_before_begin_sample = (sample-1) * elements_per_sample;
        
        for channel=1:channels

            index_result = 1;
            index_channel_begin = (channel-1) * elements_per_channel + 1;
            index_channel_end = channel * elements_per_channel;
            
            for i=1:limit_rows
                

                for j=1:limit_cols
                    patch = images(i:i+kernel_size-1, j:j+kernel_size-1, channel, sample);

                    
                    vectorized(index_result, index_channel_begin:index_channel_end, sample) = patch(:)';
                    index_result = index_result + 1;
                end

            end
        end

    end

end