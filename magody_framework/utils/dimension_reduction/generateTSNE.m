function history = generateTSNE(features_matrix, classes, options, verbose_level)
    % dir should end in '/', example: figures/
    limit_samples = options('limit_samples');
    plot_point_size = options('plot_point_size');
    include3D = options('include3D');
    dir = options('dir');
    save = options('save');

    classes_unique = unique(classes);
    num_classes_unique = length(classes_unique);
    len_data = length(classes);

    different_colors = [...
        0.5, 0, 0; 0, 1, 0; 0, 0, 1; ...
        0.7, 0, 1; 0, 0.8, 1; 1, 1, 0; ...
        1, 0.64, 0; 0, 0, 0; 0.5, 0.5, 0.5; 1, 0, 1; ...
        ];
    
    random_colors = cell([1, num_classes_unique]);
    
    
    for i=1:num_classes_unique
        if i <= length(different_colors)
            random_colors{i} = different_colors(i, :);
        else
            random_colors{i} = rand([1 3]);
        end
    end
    classes_colors = containers.Map(classes_unique, random_colors);
    

    
    colors = zeros([len_data, 3]);
    
    
    for i=1:len_data
        class_name = classes{i};
        colors(i, :) = classes_colors(class_name);
    end

    %{
    algorithms = [ ...
        struct('distance', 'euclidean','plot', struct('title', 'Euclidean')), ...
        
        struct('distance', 'mahalanobis','plot', struct('title', 'Mahalanobis')), ...
        struct('distance', 'chebychev','plot', struct('title', 'Chebychev')), ...
        struct('distance', 'hamming','plot', struct('title', 'Hamming')), ...
        struct('distance', 'cosine','plot', struct('title', 'cosine')), ...
     ];
    %}
    algorithms = options('algorithms');
    
    
    len_algorithms = length(algorithms);

    if verbose_level > 0
        fprintf("TSNE for %d algorithms\n", len_algorithms);
    end
    % plot_rows = ceil((len_algorithms)/2);
    % plot_cols = floor((len_algorithms+1)/2);

    fig_count = 1;
    
    limit = min(limit_samples, len_data);

    reduced_features_matrix = features_matrix(1:limit, :);
    reduced_classes = classes(1:limit, :);
    reduced_colors = colors(1:limit, :);
    
    
    history = containers.Map();

    for i=1:len_algorithms
        
        summary = struct();
        
        algorithm_title = string(algorithms(i).plot.title);
        
        figure(fig_count);
        [Y_2D, loss_2D] = tsne(reduced_features_matrix,'Algorithm','exact', ...
                 'Distance',algorithms(i).distance, 'NumDimensions',2);

        % subplot(plot_rows, plot_cols,i)
        gscatter(Y_2D(:,1),Y_2D(:,2), reduced_classes, different_colors, '.', plot_point_size)
        title(algorithm_title)


        if save
            saveas(gcf, dir + algorithm_title  + "_2D.png");
        end
        
        fig_count = fig_count + 1;
        summary.loss_2D = loss_2D;
        
        if include3D

            figure(fig_count+1);
            [Y_3D, loss_3D] = tsne(reduced_features_matrix,'Algorithm','exact', ...
                     'Distance',algorithms(i).distance, 'NumDimensions',3);

            v = double(categorical(classes));
            % subplot(plot_rows, plot_cols,i)
            scatter3(Y_3D(:,1),Y_3D(:,2),Y_3D(:,3),plot_point_size, reduced_colors,'filled')
            title(algorithm_title)
            
            if save
                saveas(gcf, dir + algorithm_title  + "_3D.png");
            end
            
            summary.loss_3D = loss_3D;
            
            fig_count = fig_count + 1;
        end
        
        history(algorithm_title) = summary;

        % fprintf('2-D embedding has loss %g, and 3-D embedding has loss %g.\n', loss_2D, loss_3D)


    end




end

