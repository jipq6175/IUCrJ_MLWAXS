# deep neural net for cnn waxs, 1D convolution

using Mocha, JLD
currdir = pwd();
datadir = "F:\\Yen\\ShapeData";
cd(datadir);

# It seems like that the training is very sensitive to the initialization of the random number
# Random seed
#srand(12345679);
# try
#     rm("snapshots-cpu", recursive=true);
#     info("snapshots-cpu removed");
# end
#--- I/O
dimin = 701;
dimout = 51^3;

#--- hyper-parameters

number_of_layers = 3;
number_of_conv_filters = 2^8;
# convolution and pooling
conv_filter_size = 2^3;
conv_stride = 2^0;
pool_filter_size = 2^2;
pool_stride = 2^0;

# inner product layers
# ip1_dimout = 2^8;
ipf_dimout = 2^8;

# batch size
batchsize = 2^8;


#--- neural networks
ConvolutionLayers = Vector{Mocha.ConvolutionLayer}(number_of_layers);
PoolingLayers = Vector{Mocha.PoolingLayer}(number_of_layers);
InnerProductLayers = Vector{Mocha.InnerProductLayer}(number_of_layers);

datain = AsyncHDF5DataLayer(name="train-data", source="data_train.txt", batch_size=batchsize, shuffle=true);

for i = 1:number_of_layers

    # Using the name variables
    conv_name = "conv$i";
    pool_name = "pool$i";
    ip_name = "ip$i";
    conv_bottoms = Symbol("null");
    ip_bottoms = Symbol("null");

    conv_bottoms = i == 1? Symbol("data"): Symbol("pool$(i-1)");
    ip_bottoms = i == 1? Symbol("pool$number_of_layers"): Symbol("ip$(i-1)");

    ConvolutionLayers[i] = ConvolutionLayer(name=conv_name, n_filter=number_of_conv_filters, kernel=(conv_filter_size, 1), stride=(conv_stride, 1), neuron=Neurons.Sigmoid(), bottoms=[conv_bottoms], tops=[Symbol(conv_name)]);

    PoolingLayers[i] = PoolingLayer(name=pool_name, kernel=(pool_filter_size, 1), stride=(pool_stride, 1), bottoms=[Symbol(conv_name)], tops=[Symbol(pool_name)]);


    InnerProductLayers[i] = InnerProductLayer(name=ip_name, output_dim=ipf_dimout, neuron=Neurons.ReLU(), bottoms=[ip_bottoms], tops=[Symbol(ip_name)]);

end

# The three fully connected layers are enough I believe
# final 2 innerproduct layers
ipf = InnerProductLayer(name="ipf", output_dim=dimout, neuron=Neurons.ReLU(), bottoms=[Symbol("ip$number_of_layers")], tops=[:ipf]);



# Loss layer
loss = SquareLossLayer(name="loss", bottoms=[:ipf, :label]);


# Set up common layers
common_layers = Vector{Mocha.Layer}(0);
for i = 1:number_of_layers
    append!(common_layers, [ConvolutionLayers[i]; PoolingLayers[i]]);
end
append!(common_layers, [InnerProductLayers...; ipf]);


#--- backend
backend = DefaultBackend();
init(backend);

#--- networks
net = Net("CNNWAXS", backend, [datain; common_layers...; loss]);

#--- training methods and parameters
exp_dir = "snapshots-cnnswaxs0318-$(Mocha.default_backend_type)";
maxiter = 100000;
learning_rate = 1e-5;

method = SGD();
params = make_solver_parameters(method, max_iter=maxiter, regu_coef=0.001, mom_policy=MomPolicy.Fixed(0.9), lr_policy=LRPolicy.Inv(learning_rate, 0.0001, 0.75), load_from=exp_dir);
solver = Solver(method, params);

#--- test/validation network
input_validate = HDF5DataLayer(name="test-data", source="data_validate.txt", batch_size=5*batchsize);
accuracy = SquareLossLayer(name="validation-accuracy", bottoms=[:ipf, :label]);
validation_net = Net("CNNSWAXS-validate", backend, [input_validate, common_layers..., accuracy]);

#--- Coffee Lounge
stat_interval = 100;
setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter = stat_interval);

#--- Coffee break
summary_interval = 10;
snapshot_interval = Int64(maxiter/40);
validation_interval = 100;

add_coffee_break(solver, TrainingSummary(), every_n_iter=summary_interval);
add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=snapshot_interval);
add_coffee_break(solver, ValidationPerformance(validation_net), every_n_iter=validation_interval);


# Print net
print_with_color(:light_magenta, net);
print_with_color(:light_cyan, validation_net);


#--- Solve
solve(solver, net);

#--- shut down
destroy(net);
destroy(validation_net);
shutdown(backend);
