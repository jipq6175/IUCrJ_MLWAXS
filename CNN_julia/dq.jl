using HDF5;




# Each .hdf5 file contains 64 structures;
# size(file["data"]) = (701, 1, 1, 64);
# size(file["label"]) = (51*51*51, 64);
function collect_data(sourcedir::String, targetdir::String; batchsize::Int64=128)


    # Getting data as .hdf5 from all three models
    folders = ["models"; "ModelNet40"; "ShapeNetCore.v2"; "ModelNetFull"];
    batch_size = batchsize;
    n_file::Int64 = 1;
    n_nan::Int64 = 0;
    prefix = "data_train";

    for folder in folders

        filelist = Vector{String}(0);
        for (root, dirs, files) in walkdir(joinpath(sourcedir, folder))
            println("Getting files in $root ...");
            for file in files
                push!(filelist, joinpath(root, file));
            end
        end
        filelist = filelist[endswith.(filelist, ".dat")];

        data = zeros(701, 1, 1, batch_size);
        label = zeros(51^3, batch_size);
        n_goodfile::Int64 = 0;

        for j = 1:length(filelist)

            # Read the .dat file
            swaxs = readdlm(filelist[j]);

            # NaN check
            if length(find(isnan.(swaxs))) != 0
                warn("NaN detected in $(filelist[j])!!!");
                n_nan = n_nan + 1;
                continue;
            else
                n_goodfile = n_goodfile + 1;
            end

            # 128 data points per .hdf5 file
            index = n_goodfile % batch_size;
            index == 0? index = batch_size: nothing;
            ## println("$index");

            # Read the binvox file and do the permutations
            vox = readvox(replace(filelist[j], ".dat", ".binvox"));
            # Save some calculations
            vec = log10.(swaxs[:, 2]); ## + 2 * log10(length(find(vox.rawdata)));

            # no permutation
            label[:, index] = vox.rawdata + 0.0;
            data[:, 1, 1, index] = vec;


            if (index == batch_size) || (j == length(filelist))
                h5filename = joinpath(targetdir, "$(prefix)_$(n_file).hdf5");
                h5open(h5filename, "w") do file
                    write(file, "data", data, "label", label);
                end
                println("$h5filename saved successfully.");
                n_file = n_file + 1;
                data = zeros(701, 1, 1, batch_size);
                label = zeros(51^3, batch_size);
            end
        end

        info("$(folder) extraction done!!");
    end

    datalist = readdir(targetdir);
    datalist = datalist[endswith.(datalist, "hdf5")];
    hdf5train = open(joinpath(targetdir, "data_train.txt"), "w");
    hdf5validate = open(joinpath(targetdir, "data_validate.txt"), "w");
    for i = 1:length(datalist)
        rand() >= 0.05? write(hdf5train, "$(datalist[i])\n"): write(hdf5validate, "$(datalist[i])\n");
    end
    close(hdf5train);
    close(hdf5validate);

    println("----- Summary -----");
    warn("$n_nan files contain NaN in the data, and they were skipped..");
    info("$(length(datalist)) .hdf5 files saved.. ");
end




function collect_augment_data(sourcedir::String, targetdir::String; batchsize::Int64=96)

    # batch size must be the multiple of 6due to prmutation
    batchsize % 6 !=0? error("Batch Size must be multiple of 6."): nothing;

    # Getting data as .hdf5 from all three models
    folders = ["models"; "ModelNet40"; "ShapeNetCore.v2"];
    perms = [[1;2;3] [1;3;2] [2;1;3] [2;3;1] [3;1;2] [3;2;1]];
    batch_size = batchsize;
    n_file::Int64 = 1;
    n_nan::Int64 = 0;
    prefix = "data_train";

    for folder in folders

        filelist = Vector{String}(0);
        for (root, dirs, files) in walkdir(joinpath(sourcedir, folder))
            info("Getting files in $root ...");
            for file in files
                push!(filelist, joinpath(root, file));
            end
        end
        filelist = filelist[endswith.(filelist, ".dat")];

        data = zeros(701, 1, 1, batch_size);
        label = zeros(51^3, batch_size);
        n_goodfile::Int64 = 0;

        for j = 1:length(filelist)

            # Read the .dat file
            swaxs = readdlm(filelist[j]);

            # NaN check
            if length(find(isnan.(swaxs))) != 0
                warn("NaN detected in $(filelist[j])!!!");
                n_nan = n_nan + 1;
                continue;
            else
                n_goodfile = n_goodfile + 6;
            end

            # 64 data points per .hdf5 file
            index = n_goodfile % batch_size;
            index == 0? index = batch_size: nothing;
            ## println("$index");

            # Read the binvox file and do the permutations
            vox = readvox(replace(filelist[j], ".dat", ".binvox"));
            # Save some calculations
            vec = log10.(swaxs[:, 2]); ## + 2 * log10(length(find(vox.rawdata)));
            for k = 1:6
                label[:, index - 6 + k] = reshape(permutedims(reshape(vox.rawdata, (51, 51, 51)), perms[:, k]), 51^3) + 0.0;
                data[:, 1, 1, index - 6 + k] = vec;
            end

            if (index == batch_size) || (j == length(filelist))
                h5filename = joinpath(targetdir, "$(prefix)_$(n_file).hdf5");
                h5open(h5filename, "w") do file
                    write(file, "data", data, "label", label);
                end
                info("$h5filename saved successfully.");
                n_file = n_file + 1;
                data = zeros(701, 1, 1, batch_size);
                label = zeros(51^3, batch_size);
            end
        end

        info("$(folder) extraction done!!");
    end

    datalist = readdir(targetdir);
    datalist = datalist[endswith.(datalist, "hdf5")];
    hdf5train = open(joinpath(targetdir, "data_train.txt"), "w");
    hdf5validate = open(joinpath(targetdir, "data_validate.txt"), "w");
    for i = 1:length(datalist)
        rand() >= 0.05? write(hdf5train, "$(datalist[i])\n"): write(hdf5validate, "$(datalist[i])\n");
    end
    close(hdf5train);
    close(hdf5validate);

    println("----- Summary -----");
    warn("$n_nan files contain NaN in the data, and they were skipped..");
    info("$(length(datalist)) .hdf5 files saved.. ");
end


# Modify the existing jld files
function label_normalize(sourcedir::String, targetdir::String)

    !isdir(sourcedir)? error("$sourcedir not found."): nothing;
    !isdir(targetdir)? mkdir(targetdir): nothing;

    h5list = readdir(sourcedir);
    h5list = h5list[endswith.(h5list, ".hdf5")];
    h5list_read = joinpath.(sourcedir, h5list);
    h5list_write = joinpath.(targetdir, h5list);
    n = length(h5list);
    for i = 1:n

        data = h5read(h5list_read[i], "data");
        label = h5read(h5list_read[i], "label");

        n_vox = sum(label, 1);
        label = label ./ n_vox;

        h5open(h5list_write[i], "w") do file
            write(file, "data", data, "label", label);
        end

        println("$(h5list_read[i]) normalized to $(h5list_write[i]).");
    end

    datalist = readdir(targetdir);
    datalist = datalist[endswith.(datalist, "hdf5")];
    hdf5train = open(joinpath(targetdir, "data_train.txt"), "w");
    hdf5validate = open(joinpath(targetdir, "data_validate.txt"), "w");
    for i = 1:length(datalist)
        rand() >= 0.05? write(hdf5train, "$(datalist[i])\n"): write(hdf5validate, "$(datalist[i])\n");
    end
    close(hdf5train);
    close(hdf5validate);

    println("----- Summary -----");
    println("$(length(datalist)) .hdf5 files saved.. ");
end
