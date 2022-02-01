function [lambda, v, iter] = power_deflation_rec(A, M, tol, nmax, x0)
% PRE-conditions: 
% - A is a square matrix of size n-by-n
% - M > 0 is the number of largest eigenvalues of A that we want to find
% - tol, nmax, x0 are optional parameters used to apply the iterative
% procedure power_method to find the largest eigenvalue of A and the
% associated eigenvector. Furthermore the size of x0 is n-by-1, a column
% unitary vector
%
% POST-conditions:
% - lambda is a vector M-by-1 with the largest in modulus M eigenvalues of A
% - v is a matrix n-by-M where the columns are the associated eigenvectors
%
% Among the code we will place comments to demonstrate the correctness of
% the function with respect to the PRE and POST conditions stated above.

% Initialization
if M < 1
    error('M must be >= 1');
end
[n, m] = size(A);
if n ~= m
    error('Matrix A is not square');
end
if nargin == 2
    tol = 1e-16;
    nmax = 100000;
    x0 = ones(n, 1);
end
x0 = x0 / norm(x0);
% The PRE-condition of the function power_method is respected.
[lambda_1, v_1, iter_1] = power_method(A, tol, nmax, x0);
% Therefore the POST-condition hold since we assume the correctness of the
% function:
% - lambda_1 is the eigenvector with highest modulus of A
% - v_1 is the associated eigenvector
if M == 1
    % BASE CASE (M=1)
    % lambda_1 is the highest eigenvalue of A and v_1 is n-by-1, the 
    % associated eigenvector. This condition match exactly the 
    % POST-condition of power_deflation_rec, so no more operations are 
    % needed.
    lambda = lambda_1;
    v = v_1;
    iter = iter_1;
else
    % INDUCTIVE CASE (M > 1)
    % now we deflate the matrix A obtaining B of size (n-1)-by(n-1), then
    % we find the other M-1 eigenvalues and eigenvectors of B using
    % inductive properties of this recursive function
    i = find(abs(v_1) == max(abs(v_1)), 1);
    if(i ~= 1)
        B(1:i-1, 1:i-1) = A(1:i-1,1:i-1)-v_1(1:i-1)/v_1(i)*A(i,1:i-1);
        if(i ~= n)
            B(i:n-1,1:i-1) = A(i+1:n,1:i-1)-v_1(i+1:n)/v_1(i)*A(i,1:i-1);
            B(1:i-1,i:n-1) = A(1:i-1,i+1:n)-v_1(1:i-1)/v_1(i)*A(i,i+1:n);
        end
    end
    if(i ~= n)
        B(i:n-1,i:n-1) = A(i+1:n,i+1:n)-v_1(i+1:n)/v_1(i)*A(i,i+1:n);
    end
    % The recursive PRE-Condition hold:
    % - B is a square matrix (n-1)-by-(n-1)
    % - M-1>0 (M>1) is the number of eigenvalues of B to find 
    [lambda_next, v_next, iter_next] = power_deflation_rec(B, M-1);
    % Therefore, by inductive assumption, the recursive POST-condition hold:
    % - lambda_next is a vector (M-1)-by-1 with the M-1 highest in modulus
    % eigenvalues of B
    % - v_next is a matrix (n-1)-by-(M-1) whose columns are the associated
    % eigenvectors
    %
    % now we need to transform the eigenvector returned by the previous
    % call in order to satisfy the general POST-condition of this function,
    % then by induction the correctness is fulfilled.
    %
    % because of the deflation properties we do not need to apply
    % any operations to eigenvalues, we just respect the size of the vector
    % lambda imposed by the POST-condition
    lambda = [lambda_1; lambda_next];
    iter = [iter_1; iter_next];
    v = zeros(n, M);
    v(:, 1) = v_1;
    % now we transform eigenvectors returned by the recursive call to
    % dimensionality n, while we satisy the constraints on the matrix v
    % stated in the POST-condition
    for j = 1:(M-1) % scanning columns of v_next
        w_prime = v_next(:, j);
        w = zeros(n, 1);
        if(i ~= 1)
            w(1:i-1) = w_prime(1:i-1);
        end
        w(i) = 0;
        if(i ~= n)
            w(i+1:n) = w_prime(i:n-1);
        end
        v(:, j+1) = w;
    end
end
end