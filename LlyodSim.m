%Distribution Parameters
mu = 0;
sigma = 1;
n = 5000;

%Llyod-Max Parameters
eps = 0.01;

%Run simulations
[cb1, D1, m1] = lloydSim(mu, sigma, n, 1, eps);
[cb2, D2, m2] = lloydSim(mu, sigma, n, 2, eps);
[cb4, D4, m4] = lloydSim(mu, sigma, n, 4, eps);
[cb8, D8, m8] = lloydSim(mu, sigma, n, 8, eps);

%Plot distortion results
X = [1, 2, 4, 8];
Y = [D1, D2, D4, D8];
CB = {cb1; cb2; cb4; cb8};

figure
plot(X, Y);
xlabel('Codebook Size (n)');
ylabel('Average Distortion');
title('Distortion for n-length Codebook');

%Output optimal codebooks
disp('Optimal Codebooks:');
for i = 1:size(X, 2)
    fprintf('n=%d: ', X(i));
    disp(CB{i});
end




function [cb, D, m] = lloydSim(mu, sigma, n, cb_n, eps)
    %Init test data and codebook
    T = sort(normrnd(mu, sigma, [1, n]));

    cb_1 = linspace(-1, 1, cb_n);
    [cb, D, m] = llyodRecursive(T, cb_1, eps, 0, 1);
end

function [cb, D, m] = llyodRecursive(T, cb_m, eps, D_last, m)
    
    %Partition using NNC:------------------------------------------------
    %Sort codebook for easy test data partitioning
    cb_m = sort(cb_m);
    
    %Create bins for each value of codebook
    R = {};
    R{size(cb_m, 2)} = [];
    
    %Having CB and T sorted allows very easy assignment to bins:
    %Loop through data points and bins, if next bin has less distortion
    %increase bin number, do for every data point
    i = 1;
    for d = 1:size(T, 2)
        while true
            %Last bin condition
            if i == size(cb_m, 2)
                R{i}(end+1) = T(d);
                break
            end
            if abs(cb_m(i)-T(d)) < abs(cb_m(i+1)-T(d))
                R{i}(end+1) = T(d);
                break
            else
                i = i+1;
            end
        end
    end
    %---------------------------------------------------------------------
    
    
    %If first loop: need to calculate intial codebook distortion for D_last
    if m == 1
        D_last = 0;
        for i = 1:size(cb_m, 2)
            for d = 1:size(R{i}, 2)
                D_last = D_last + R{i}(d);
            end
        end
        D_last = D_last / size(T, 2);
    end
    
    
    %Find optimal codebook using partition R and CC: ---------------------
    cb = zeros(1, size(cb_m, 2));
    for i = 1:size(cb_m, 2)
        sum = 0;
        for d = 1:size(R{i}, 2)
            sum = sum + R{i}(d);
        end
        cb(i) = sum / size(R{i}, 2); %Possible for NAN if no numbers in bin
    end
    %---------------------------------------------------------------------
    
    %Check if change in distortion < eps (Optimal Codebook Found):--------
    %Calculate new avg distortion
    D = 0;
    for i = 1:size(cb, 2)
        for d = 1:size(R{i}, 2)
            D = D + abs(cb(i) - R{i}(d));
        end
    end
    D = D / size(T, 2);
    
    m = m+1;
    %Compare distortion change
    if (D - D_last) / D >= eps
        [cb, D, m] = llyodRecursive(T, cb, eps, D, m);
    end
    %---------------------------------------------------------------------
end