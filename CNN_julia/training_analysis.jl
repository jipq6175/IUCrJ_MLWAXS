using Plots, JLD
gr();
include("file_operations.jl");
statfile = "F:\\Yen\\ShapeData\\snapshots0221-cpu\\statistics.jld";
objmat, valmat = training_stat(statfile);
obj_smooth = [mean(objmat[(i-5):(i+4), 2]) for i = 6:(size(objmat,1)-4)];


objplt = plot(objmat[1:end,1], objmat[1:end,2], lw=0.5, alpha=0.25);
plot!(objmat[6:end-4, 1], obj_smooth, lw=2.5);
plot!(size=(1000, 750), dpi=600, legend=false);
ylabel!("Δ");
xlabel!("Iterations")
savefig(objplt, "Training.png");


val_smooth = [mean(valmat[(i-3):(i+2), 2]) for i = 4:(size(valmat,1)-2)];
valplt = plot(valmat[1:end,1], valmat[1:end,2], lw=0.5, alpha=0.25);
plot!(valmat[4:end-2, 1], val_smooth, lw=2.5);
plot!(size=(1000, 750), dpi=600, legend=false);
ylabel!("Δ");
xlabel!("Iterations")
savefig(valplt, "Validating.png");


## Testing the prection function
