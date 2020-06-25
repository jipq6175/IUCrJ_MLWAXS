# deep neural net for cnn waxs, 1D convolution

using Mocha, JLD

# Random seed
#srand(12345679);
try
    rm("snapshots-cpu", recursive=true);
    info("snapshots-cpu removed");
end
#--- I/O
dimin = 701;
dimout = 51;

# batch size
batchsize = 2^13;

# activation function = ReLU()


#--- construct the layers of the network
datain = AsyncHDF5DataLayer(name="train-data", source="models_train.txt", batch_size=batchsize, shuffle=true);

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
exp_dir = "snapshots-$(Mocha.default_backend_type)";
maxiter = 10000;
learning_rate = 5e-6;

method = SGD();
params = make_solver_parameters(method, max_iter=maxiter, regu_coef=0.0005, lr_policy=LRPolicy.Inv(learning_rate, 0.0001, 0.75), load_from=exp_dir);
solver = Solver(method, params);

#--- test/validation network
input_validate = HDF5DataLayer(name="test-data", source="models_validate.txt", batch_size=5*batchsize);
accuracy = SquareLossLayer(name="validation-loss", bottoms=[:ip5, :label]);
validation_net = Net("CNNSWAXS-validate", backend, [input_validate, common_layers..., accuracy]);

#--- Coffee Lounge
stat_interval = 10;
setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter = stat_interval);

#--- Coffee break
summary_interval = 5;
#snapshot_interval = maxiter;
validation_interval = 50;

add_coffee_break(solver, TrainingSummary(), every_n_iter=summary_interval);

# add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=snapshot_interval);

add_coffee_break(solver, ValidationPerformance(validation_net), every_n_iter=validation_interval);

#--- solving neural network
solve(solver, net);


#--- shut down
destroy(net);
destroy(validation_net);
shutdown(backend);
