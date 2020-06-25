using Plots, HDF5, JLD, StatsBase;

#
gr();


# scanning across all the waxs curves
dir = "F:\\Yen\\ShapeData_Permuted";
function plot_all(dir::String, skip::Int64)
    datalist = readdir(dir);
    datalist = joinpath.(dir, datalist[find(endswith.(datalist, "hdf5"))]);

    nhdf5 = length(datalist);
    # skip = 6;
    q = collect(0.0:0.002:1.4);
    plot(size=(1000, 750));

    style = [:auto, :solid, :dash, :dot, :dashdot, :dashdotdot];
    n = 1;
    for i = 1:nhdf5

        info("Loading data: $(datalist[i]) ...");
        curve = h5read(datalist[i], "data")[:, 1, 1, :];
        ncurve = Int64(size(curve, 2) / skip);

        for j = 1:ncurve
            n % 160 == 0? plot(size=(1000, 750)): nothing;
            plot!(q, curve[:, 6*(j-1)+1], lw=5.0, size=(1000, 750), legend=false, ls=style[rand(collect(1:6))], alpha=0.3);
            ylims!(-6.0, 0.5);
            gui();
            sleep(0.05);
            n = n + 1;
        end
    end
end


# Randomly choose a portion of data into a new dir
function reduce_data_set(sourcedir::String, targetdir::String; folds::Int64=4)

    datalist = readdir(sourcedir);
    datalist = datalist[find(endswith.(datalist, ".hdf5"))];
    n = length(datalist);
    transferlist = Vector{String}(0);

    for i = 1:n
        if rand() < 1.0/folds
            cp(joinpath(sourcedir, datalist[i]), joinpath(targetdir, datalist[i]));
            push!(transferlist, datalist[i]);
            info("$(datalist[i]) was transferred.");
        end
    end

    hdf5train = open(joinpath(targetdir, "data_train.txt"), "w");
    hdf5validate = open(joinpath(targetdir, "data_validate.txt"), "w");
    for i = 1:length(transferlist)
        rand() >= 0.05? write(hdf5train, "$(transferlist[i])\n"): write(hdf5validate, "$(transferlist[i])\n");
    end
    close(hdf5train);
    close(hdf5validate);
    info("Data reduction done: From $(length(datalist)) to $(length(transferlist))..");
end


# Load the statistics of training and validation
function training_stat(filename::String)

    stat = load(filename, "statistics");
    val = sort!(collect(stat["validation-accuracy-square-loss"]), by = x -> x[1]);
    obj = sort!(collect(stat["obj_val"]), by = x -> x[1]);

    valmat = zeros(length(val), 2);
    [valmat[i, :] = [val[i][1] val[i][2]] for i = 1:length(val)];
    objmat = zeros(length(obj), 2);
    [objmat[i, :] = [obj[i][1] obj[i][2]] for i = 1:length(obj)];

    return objmat, valmat;
end


# Clean unnecessary files, leave .hdf5, .jld, .dat, .binvox, .obj, .pdb
function clean_dir(dir::String; wanted::Vector{String}=[".hdf5"; ".jld"; ".dat"; ".binvox"; ".obj"; ".pdb"; ".off"])

    for (root, dirs, files) in walkdir(dir)
        println("Getting files in $root ...");
        for file in files
            if length(find(endswith.(file, wanted))) == 0
                println("Removing: $(file) ..");
                rm(joinpath(root, file));
            end
        end
    end
end


# Remove empty directories
function clean_modelnet(dir::String)

    for (root, dirs, files) in walkdir(dir)
        println("Getting files in $root ...");
        for file in files

            if filesize(joinpath(root, file)) == 0
                println("$(file): $(filesize(joinpath(root, file)))");
                info("Removing directory: $(root) ..");
                rm(root, recursive=true);
            end
        end
    end

end


function debye_distance(mat::Matrix{Float64})
    n = size(mat, 1);
    coor = Array{Float64}(n, n, 3);
    for i = 1:3
        M = reshape(repmat(mat[:, i], n), (n, n));
        coor[:, :, i] = (M - M').^2;
    end
    r = sqrt.(sum(coor, 3)[:, :, 1]);
    return r;
end
