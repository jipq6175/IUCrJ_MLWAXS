
# Dependencies
using Mocha, JLD


# System Parameters
currdir = pwd();
datadir = "F:\\Yen\\ShapeData";
cd(datadir);


# Neural Network architecture
dimin = 701;
dimout = 51^3;

number_of_layers = 12;
ip_output_dim = 2^9;

batchsize = 2^9;


# Backend
backend = DefaultBackend();
init(backend);


# Layers
datain = AsyncHDF5DataLayer(name="train-data", source="data_train.txt", batch_size=batchsize, shuffle=true);

# InnerProductLayers = Vector{Mocha.InnerProductLayer}(number_of_layers);
#
# for i = 1:number_of_layers
#     ip_name = "ip$i";
#     ip_bottoms = Symbol("null");
#     ip_tops = Symbol(ip_name);
#
#     ip_bottoms = i == 1? Symbol("data"): Symbol("ip$(i-1)");
#
#     InnerProductLayers[i] = InnerProductLayer(name=ip_name, output_dim=ip_output_dim, neuron=Neurons.ReLU(), bottoms=[ip_bottoms], tops=[ip_tops]);
#
# end

ipf = InnerProductLayer(name="ipf", output_dim=dimout, neuron=Neurons.Sigmoid(), bottoms=[:data], tops=[:ipf]);
loss = SquareLossLayer(name="loss", bottoms=[:ipf, :label]);

common_layers = [ipf];


# Training Net
training_net = Net("IPSWAXS-TRAINING", backend, [datain; common_layers...; loss]);


# Validation Net
input_validate = HDF5DataLayer(name="test-data", source="data_validate.txt", batch_size=5*batchsize);
accuracy = SquareLossLayer(name="validation-accuracy", bottoms=[:ipf, :label]);
validation_net = Net("IPSWAXS-VALIDATE", backend, [input_validate; common_layers...; accuracy]);


# Print the networks
print_with_color(:light_magenta, training_net);
print_with_color(:light_cyan, validation_net);


# Training parameters
exp_dir = "snapshots-ipswaxs-0325-pc-$(Mocha.default_backend_type)";
maxiter = 100000;
learning_rate = 5e-6;


method = SGD();
params = make_solver_parameters(method, max_iter=maxiter, regu_coef=0.0001, mom_policy=MomPolicy.Fixed(0.9), lr_policy=LRPolicy.Inv(learning_rate, 0.0001, 0.75), load_from=exp_dir);
solver = Solver(method, params);


# Coffee Lounge
stat_interval = 25;
setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter = stat_interval);


# Coffee break
summary_interval = 10;
snapshot_interval = Int64(maxiter/20);
validation_interval = 100;

add_coffee_break(solver, TrainingSummary(), every_n_iter=summary_interval);
add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=snapshot_interval);
add_coffee_break(solver, ValidationPerformance(validation_net), every_n_iter=validation_interval);


# Solve
solve(solver, training_net);


# shut down
destroy(training_net);
destroy(validation_net);
shutdown(backend);
cd(currdir);
