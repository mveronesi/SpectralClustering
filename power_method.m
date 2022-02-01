function [lambda, x, iter] = power_method(A, tol, nmax, x0)
% PRE-conditions:
% - A is a square matrix of size n-by-n
% - tol, nmax, x0 are parameters used to apply the power method.
% Furthermore x0 has size n-by-1 and it is a unitary vector.
%
% POST-conditions:
% - lambda is the eigenvalues with the highest modulus of A
% - x is the associated eigenvector
% - iter is the number of iteration needed to reach the tolerance
%
% We assume the correctness of this function and do NOT demonstrate it
% with respect to the PRE and POST conditions stated above.

    pro = A*x0;
    lambda = x0'*pro ;
    err = tol*abs(lambda)+1;
    iter = 0;
    while (err > tol*abs ( lambda )) && (iter <= nmax)
        x = pro;
        x = x/norm(x);
        pro = A*x;
        lambdanew = x'*pro ;
        err = abs(lambdanew-lambda);
        lambda = lambdanew ;
        iter = iter + 1;
    end
end