
# Compare calculation of debye formula using CPU and GPU
using CLArrays

function debye_cpu(m::Matrix{T}, q::T) where T<:Real
    n = size(m, 1);
    coor = Array{Float64}(n, n, 3);
    for i = 1:3
        M = reshape(repmat(m[:, i+1], n), (n, n));
        coor[:, :, i] = (M - M').^2;
    end
    r = sqrt.(sum(coor, 3)[:, :, 1]);

    mat = q*r;
    sinqr = sin.(mat)./(mat);
    sinqr[1:(n+1):n*n] = 1.0;
    Ii = sum(sinqr);
    return Ii;
end


function debye_gpu(m::Matrix{T}, q::T) where T<:Real
    n = size(m, 1);
    coor = zeros(CLArray{Float64}, n, n, 3);
    r = zeros(CLArray{Float64}, n, n);

    for i = 1:3
        M = reshape(repmat(m[:, i+1], n), (n, n));
        coor[:, :, i] = (M - M').^2;
    end

    r = sqrt.(sum(coor, 3)[:, :, 1]);

    r = q .* r;
    r = sin.(r) ./ r;
    r[1:(n+1):n*n] = 1.0;
    Ii = sum(r);
    return Ii;
end
