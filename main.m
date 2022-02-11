% Initialization
clear; close all; clc;
format long;
load('X.mat');
sigma = 1; tol = 1e-8;
similarity = @(x, y) exp(-norm(x-y, 2).^2 ./ sigma.^2);
% parameters for clustering results plots
symbols = ['.', '+', 'o'];
colors = ['k', 'r', 'b'];

for K = [13, 40]
    disp(['############ K = ', num2str(K), ' #############']);
    % Generating similarity matrix between points
    [n, ~] = size(X);
    S1 = zeros(n, n);
    for i=1:(n-1)
        p1 = X(i, :);
        for j=(i+1):n
            p2 = X(j, :);
            s = similarity(p1, p2);
            if s > tol
                S1(i,j) = s;
                S1(j,i) = s;
            end
        end
    end
    figure();
    spy(S1);
    title('Sparsity pattern of S');
    S2 = S1;
    % Generating similarity graph with k-nearest neighbourhood graph
    for i = 1:n
        [~, kmax_pos] = maxk(S1(i, :), K);
        S1(i, setdiff(1:n, kmax_pos)) = 0;
    end
    for j = 1:n
        [~, kmax_pos] = maxk(S2(:, j), K);
        S2(setdiff(1:n, kmax_pos), j) = 0;
    end
    max_nz = nnz(S1) + nnz(S2);
    column_indices = zeros(max_nz, 1);
    row_indices = zeros(max_nz, 1);
    values = zeros(max_nz, 1);
    nz = 0;
    for i=1:n
        for j=1:n
            if S1(i,j) > tol && S2(i,j) <= tol
                nz = nz + 1;
                column_indices(nz) = j;
                row_indices(nz) = i;
                values(nz) = S1(i,j);
            elseif S1(i,j) <= tol && S2(i,j) > tol
                nz = nz + 1;
                column_indices(nz) = j;
                row_indices(nz) = i;
                values(nz) = S2(i,j);
            elseif S1(i,j) > tol && S2(i,j) > tol
                if S1(i,j) - S2(i,j) > tol % should be equals
                    warning("Something went wrong...");
                end
                nz = nz + 1;
                column_indices(nz) = j;
                row_indices(nz) = i;
                values(nz) = S1(i,j);
            end
        end
    end
    column_indices = column_indices(1:nz);
    row_indices = row_indices(1:nz);
    values = values(1:nz);
    W = sparse(row_indices, column_indices, values, n, n);
    
    figure();
    spy(W);
    title({'Sparsity pattern of W', ['K=', num2str(K)]});
    
    figure();
    plot(graph(W), '-.dr', 'NodeLabel', {});
    title({'Similarity graph', ['K=', num2str(K)]});
    
    % Computing the normalized symmetric Laplacian
    degrees = full(sum(W, 2));
    D = spdiags(degrees, 0, n, n);

    M = 5;
    
    D_tmp = sqrt(inv(D));
    B = D_tmp * W * D_tmp;
    L_sym = speye(n) - B;
    
    % Computing eigenvalues and eigenvectors with eigs function
    tic;
    [U_orig, lambda_orig] = eigs(L_sym, M, 'smallestabs');
    elapsed = toc;
    fprintf('Elapsed time with eigs: %.2f sec\n', elapsed)
    lambda_orig = diag(lambda_orig);
    
    % Computing eigenvalues and eigenvectors with power method and deflation
    mu = max(full(sum(B, 2)));
    B_mod = B + mu .* speye(n);
    tic;
    [lambda, U, iter] = power_deflation_rec(B_mod, M);
    elapsed = toc;
    fprintf('Elapsed time with power_deflation_rec: %.2f sec\n', elapsed)
    lambda = 1-lambda+mu;
    rel_err_eigs = norm(lambda - lambda_orig, 2);
    fprintf('Error in the computation of eigenvalues: %e\n', rel_err_eigs);
    
    figure();
    bar(1:M, iter);
    title({'Number of iterations in power method', ['K=', num2str(K)]});
    xlabel('Eigenvalue');
    ylabel('Number of iterations performed');
    
    switch K
        case 13
            EIGENGAP = 3;
        case 40
            EIGENGAP = 2;
        otherwise
            warning('Eigengap not computed for this value of K, setting to 3');
            EIGENGAP = 3;
    end
    
    U = U(:, 1:EIGENGAP);

    % normalizing eigenvectors
    for j=1:EIGENGAP
        U(:,j) = U(:,j) ./ norm(U(:,j));
    end
    
    % Computating matrix U containing first m eigenvectors 
    % and normalizing its rows
    for i=1:n
        U(i,:) = U(i,:) ./ norm(U(i,:));
    end
    
    % Applying K-means for spectral clustering
    for N_CLUSTERS = [2, 3]
        
        rng(42); % setting random state for kmeans reproducibility
        idx = kmeans(U, N_CLUSTERS);
        figure();
        gscatter(X(:,1), X(:,2), idx, colors, symbols);
        title({'Kmeans result on spectral clustering', ...
            ['K=', num2str(K)], ['N\_CLUSTERS=', num2str(N_CLUSTERS)], ...
            'Eigenvectors computed with power method and deflation'});
        
        % Using spectral cluster of MATLAB.
        rng(42); % setting random state for kmeans reproducibility
        [idx, V, D] = spectralcluster(X, N_CLUSTERS);
        figure();
        gscatter(X(:,1), X(:,2), idx, colors, symbols);
        title({'Kmeans result on spectral clustering [MATLAB]', ...
            ['N\_CLUSTERS=', num2str(N_CLUSTERS)]});
        
        % Applying K-means on original points
        rng(42); % setting random state for kmeans reproducibility
        idx = kmeans(X, N_CLUSTERS);
        figure();
        gscatter(X(:,1), X(:,2), idx, colors, symbols);
        title({'Kmeans result on original points', ...
            ['N\_CLUSTERS=', num2str(N_CLUSTERS)]});

        % comparing eigenvalues computed by the function 
        % spectralcluster to those of our function
        figure();
        scatter(1:M, lambda_orig, 200, 'd', 'MarkerEdgeColor', 'black', ...
            'LineWidth', 2);
        hold on;
        scatter(1:length(D), D, 100, 'o', 'MarkerEdgeColor', 'red', ...
            'LineWidth', 2.5);
        scatter(1:M, lambda, 50, 'x', 'MarkerEdgeColor', 'blue', ...
            'LineWidth', 2);
        hold off;
        title({'Minimum eigenvalues of L\_sym', [' K=', num2str(K)], ...
            [' N\_CLUSTERS=', num2str(N_CLUSTERS)]});
        legend('Computed with eigs function', ...
            'Computed with spectralcluster function', ...
            'Computed with power method with deflation', ...
             'Location', 'northwest');     
    end
end
