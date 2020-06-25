# This file contains some neural networks for CNNSWAXS
using Mocha, JLD, HDF5

include("voxel.jl");

sourcedir = "F:\\OS\\ShareData\\Shapes";
targetdir = "C:\\Users\\Yen-Lin\\Desktop\\models";

function convolution(dir::String, iter::Int64; maxiter::Int64=10000, batchsize::Int64=2^12, learning_rate::Float64=1e-5, method=SGD())

    currdir = pwd();
    cd(dir);
    try
        rm("snapshots-$(Mocha.default_backend_type)_$iter", recursive=true);
        info("snapshots-$(Mocha.default_backend_type)_$iter removed");
    end
    #--- I/O
    dimin = 701;
    dimout = 51;

    #--- hyper-parameters

    number_of_conv_filters = 100;
    # first convolution and pooling
    conv1_filter_size = 10;
    conv1_stride = 1;
    pool1_filter_size = 4;
    pool1_stride = 2;
    # second convolution and pooling
    conv2_filter_size = 10;
    conv2_stride = 3;
    pool2_filter_size = 4;
    pool2_stride = 2;
    # third convolution and pooling
    conv3_filter_size = 5;
    conv3_stride = 1;
    pool3_filter_size = 4;
    pool3_stride = 2;

    # inner product layers
    ip1_dimout = 128;
    ip2_dimout = 64;

    # batch size
    # batchsize = 2^12;

    # activation function = ReLU()


    #--- construct the layers of the network
    datain = AsyncHDF5DataLayer(name="train-data", source="data_train.txt", batch_size=batchsize, shuffle=true);
    #norm0 = LRNLayer(name="norm0", kernel=5, scale=5e-5, power=0.75, mode=LRNMode.WithinChannel(), bottoms=[:data], tops=[:norm0]);

    # conv1 + pool1
    conv1 = ConvolutionLayer(name="conv1", n_filter=number_of_conv_filters, kernel=(conv1_filter_size, 1), stride=(conv1_stride, 1), neuron=Neurons.Sigmoid(), bottoms=[:data], tops=[:conv1]);
    pool1 = PoolingLayer(name="pool1", kernel=(pool1_filter_size, 1), stride=(pool1_stride, 1), bottoms=[:conv1], tops=[:pool1]);
    #norm1 = LRNLayer(name="norm1", kernel=5, scale=5e-5, power=0.75, mode=LRNMode.WithinChannel(), bottoms=[:pool1], tops=[:norm1]);

    # conv2 + pool2
    conv2 = ConvolutionLayer(name="conv2", n_filter=number_of_conv_filters, kernel=(conv2_filter_size, 1), stride=(conv2_stride, 1), neuron=Neurons.Sigmoid(), bottoms=[:pool1], tops=[:conv2]);
    pool2 = PoolingLayer(name="pool2", kernel=(pool2_filter_size, 1), stride=(pool2_stride, 1), bottoms=[:conv2], tops=[:pool2]);
    #norm2 = LRNLayer(name="norm2", kernel=5, scale=5e-5, power=0.75, mode=LRNMode.WithinChannel(), bottoms=[:pool2], tops=[:norm2]);

    # conv3 + pool3
    conv3 = ConvolutionLayer(name="conv3", n_filter=number_of_conv_filters, kernel=(conv3_filter_size, 1), stride=(conv3_stride, 1), neuron=Neurons.Sigmoid(), bottoms=[:pool2], tops=[:conv3]);
    pool3 = PoolingLayer(name="pool3", kernel=(pool3_filter_size, 1), stride=(pool3_stride, 1), bottoms=[:conv3], tops=[:pool3]);

    # ip1
    ip1 = InnerProductLayer(name="ip1", output_dim=ip1_dimout, neuron=Neurons.Sigmoid(), bottoms=[:pool3], tops=[:ip1]);
    # ip2
    ip2 = InnerProductLayer(name="ip2", output_dim=ip2_dimout, neuron=Neurons.Sigmoid(), bottoms=[:ip1], tops=[:ip2]);
    # ip3
    ip3 = InnerProductLayer(name="ip3", output_dim=dimout, neuron=Neurons.ReLU(), bottoms=[:ip2], tops=[:ip3]);
    # loss layer
    loss = SquareLossLayer(name="loss", bottoms=[:ip3, :label]);

    #--- backend
    backend = DefaultBackend();
    init(backend);

    #--- networks
    common_layers = [conv1; pool1; conv2; pool2; conv3; pool3; ip1; ip2; ip3];
    net = Net("CNNWAXS-train", backend, [datain; common_layers...; loss]);


    #--- training methods and parameters
    exp_dir = "snapshots-$(Mocha.default_backend_type)_$iter";
    # maxiter = 10000;
    # learning_rate = 0.001;

    #method = SGD();
    params = make_solver_parameters(method, max_iter=maxiter, regu_coef=0.0005, mom_policy=MomPolicy.Fixed(0.9), lr_policy=LRPolicy.Inv(learning_rate, 0.0001, 0.75), load_from=exp_dir);
    solver = Solver(method, params);

    #--- test/validation network
    input_validate = HDF5DataLayer(name="test-data", source="data_validate.txt", batch_size=5*batchsize);
    accuracy = SquareLossLayer(name="validation-accuracy", bottoms=[:ip3, :label]);
    validation_net = Net("CNNSWAXS-validate", backend, [input_validate, common_layers..., accuracy]);

    #--- Coffee Lounge
    stat_interval = 100;
    setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter = stat_interval);

    #--- Coffee break
    summary_interval = 1;
    snapshot_interval = maxiter;
    validation_interval = 10;

    add_coffee_break(solver, TrainingSummary(), every_n_iter=summary_interval);
    add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=snapshot_interval);
    add_coffee_break(solver, ValidationPerformance(validation_net), every_n_iter=validation_interval);

    #--- solving neural network
    solve(solver, net);

    #--- shut down
    destroy(net);
    destroy(validation_net);
    shutdown(backend);
    cd(currdir);
end


function innerproduct(dir::String, iter::Int64; maxiter::Int64=10000, batchsize::Int64=2^13, learning_rate::Float64=7.5e-6, method=SGD())
    # Random seed
    #srand(12345679);
    currdir = pwd();
    cd(dir);

    try
        rm("snapshots-$(Mocha.default_backend_type)_$iter", recursive=true);
        info("snapshots-$(Mocha.default_backend_type)_$iter removed");
    end
    #--- I/O
    dimin = 701;
    dimout = 51;

    # batch size
    # batchsize = 2^13;

    # activation function = ReLU()

    #--- construct the layers of the network
    datain = AsyncHDF5DataLayer(name="train-data", source="data_train.txt", batch_size=batchsize, shuffle=true);

    # ip
    ip1 = InnerProductLayer(name="ip1", output_dim=512, neuron=Neurons.ReLU(), bottoms=[:data], tops=[:ip1]);
    ip2 = InnerProductLayer(name="ip2", output_dim=256, neuron=Neurons.ReLU(), bottoms=[:ip1], tops=[:ip2]);
    ip3 = InnerProductLayer(name="ip3", output_dim=128, neuron=Neurons.ReLU(), bottoms=[:ip2], tops=[:ip3])
    ip4 = InnerProductLayer(name="ip4", output_dim=64, neuron=Neurons.ReLU(), bottoms=[:ip3], tops=[:ip4])
    ip5 = InnerProductLayer(name="ip5", output_dim=dimout, neuron=Neurons.ReLU(), bottoms=[:ip4], tops=[:ip5])

    # loss layer
    loss = SquareLossLayer(name="loss", bottoms=[:ip5, :label]);

    #--- backend
    backend = DefaultBackend();
    init(backend);

    #--- networks
    common_layers = [ip1; ip2; ip3; ip4; ip5];
    net = Net("CNNWAXS-train", backend, [datain; common_layers...; loss]);


    #--- training methods and parameters
    exp_dir = "snapshots-$(Mocha.default_backend_type)_$iter";
    # maxiter = 10000;
    # learning_rate = 1e-5;

    # method = SGD();
    params = make_solver_parameters(method, max_iter=maxiter, regu_coef=0.0005, lr_policy=LRPolicy.Inv(learning_rate, 0.0001, 0.75), load_from=exp_dir);
    solver = Solver(method, params);

    #--- test/validation network
    input_validate = HDF5DataLayer(name="test-data", source="data_validate.txt", batch_size=5*batchsize);
    accuracy = SquareLossLayer(name="validation-loss", bottoms=[:ip5, :label]);
    validation_net = Net("CNNSWAXS-validate", backend, [input_validate, common_layers..., accuracy]);

    #--- Coffee Lounge
    stat_interval = 10;
    setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter = stat_interval);

    #--- Coffee break
    summary_interval = 5;
    snapshot_interval = maxiter;
    validation_interval = 50;

    add_coffee_break(solver, TrainingSummary(), every_n_iter=summary_interval);

    add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=snapshot_interval);

    add_coffee_break(solver, ValidationPerformance(validation_net), every_n_iter=validation_interval);

    #--- solving neural network
    solve(solver, net);

    #--- shut down
    destroy(net);
    destroy(validation_net);
    shutdown(backend);
    cd(currdir);
end





function training_data(sourcedir::String, targetdir::String, iter::Int64)

    # Getting data as .hdf5 from all three models
    folders = ["models"; "ModelNet40"; "ShapeNetCore.v2"];
    perms = [[1;2;3] [1;3;2] [2;1;3] [2;3;1] [3;1;2] [3;2;1]];
    batch_size = 100;
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

            # Read the binvox file and do the permutations
            vox = readvox(replace(filelist[j], ".dat", ".binvox"));
            # Save some calculations
            vec = log10.(swaxs[:, 2]) + 2 * log10(length(find(vox.rawdata)));
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
                label = zeros(51^1, batch_size);
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


function train_networks(sourcedir::String, network::String; targetdir::String=pwd(), n::Int64=51^2)

    for iter = 1:n

        if isdir(joinpath(targetdir, "snapshots-cpu_$iter"))
            warn("snapshots-cpu_$iter has already existed. Skipped this iteration.");
            continue;
        else
            warn("snapshots-cpu_$iter does not exist. Proceeding...");
        end

        # Get data into the target dir
        training_data(sourcedir, targetdir, iter);

        # train
        if network == "innerproduct"
            innerproduct(targetdir, iter);
        elseif network == "convolution"
            convolution(targetdir, iter);
        else
            error("$network not recognized!!");
        end

        # remove the data
        datalist = readdir(targetdir);
        datalist = datalist[startswith.(datalist, "data")];
        rm.(datalist, force=true);

        info("$iter training done!!");
    end
end
