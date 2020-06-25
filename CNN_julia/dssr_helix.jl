# clean the data of helical parameters for regression training


using JLD, HDF5, LinearAlgebra, Statistics, DelimitedFiles, Distributions


mutable struct DSSR
    bps::Int64;
    helix_form::Vector{Char};
    helical_rise::Tuple{Float64, Float64};
    helical_radius::Tuple{Float64, Float64};
    helical_extend::Float64;
    h_rise::Vector{Float64};
    h_twist::Vector{Float64};
end


mutable struct Analysis{T<:Real}
    data::Matrix{T};
    pool::Matrix{T};
    qstart::T;
    ndata::Matrix{T};
    nfit::Matrix{T};
    chi_pre::Vector{T};
    dc::Vector{T};
    survivors::Matrix{Int64};
    fitness::Vector{T};
end






# function that save the data to the specified directory
function helix_collect_data(parameter::String; datsourcedir::String="F:\\Yen\\HJH_KCl_Hydration", dssrsourcedir::String="G:\\My Drive\\13. Julia\\05. WAXSiS\\x3dnadata", targetdir::String="F:\\Yen\\DuplexData_Full")


    dict = Dict([("helical_extend", :helical_extend), ("helical_form_ratio", :helix_form), ("helical_hrise", :helical_rise), ("helical_htwist", :h_twist), ("helical_radius", :helical_radius)]);

    categories = ["30"; "50"; "200"; "500"];

    savedir = joinpath(targetdir, parameter);

    function clean(x::DSSR, parameter::String; dict::Dict{String, Symbol}=dict)

        fd = dict[parameter];
        tmp = getfield(x, fd);
        data = -1.0;
        if parameter == "helical_extend"
            data = tmp;
        elseif parameter == "helical_form_ratio"
            data = sum(tmp .== 'A') / length(tmp);
        elseif parameter == "helical_hrise"
            data = tmp[1];
        elseif parameter == "helical_htwist"
            data = sum(tmp);
        elseif parameter == "helical_radius"
            data = tmp[1];
        else
            error("Parameter $parameter is not recognized. ");
        end
        return data;
    end

    # Data augmentation by incorportating the constant c for buffer subtraction and dark current
    #augc = [0.0; 1.0e5; 1.5e5; 2.0e5; 2.5e5; 3.0e5];

    # Add oversampling options


    for c in categories

        println("Category $c :");

        # Load the dssr data
        dssrfile = "x3dna$c.jld";
        x3 = load(joinpath(dssrsourcedir, dssrfile), "dssr$c");
        n = length(x3)
        label = zeros(n);
        [label[i] = clean(x3[i], parameter) for i = 1:n];

        # Load the .dat data
        datdir = joinpath(datsourcedir, "KCl$(c)mM");
        datlist = readdir(datdir);
        datlist = datlist[endswith.(datlist, ".dat")];


        for datfile in datlist

            data = readdlm(joinpath(datdir, datfile));
            data = log10.(data[:, 2:end]);
            @info("datasize = $(size(data, 2)); labelsize = $n. ");

            h5filename = joinpath(savedir, replace(datfile, ".dat" => ".hdf5"));
            h5open(h5filename, "w") do file
                write(file, "data", data, "label", label);
            end
            println("$h5filename written successfully.");
        end
    end

    trainlist = readdir(savedir);
    trainlist = trainlist[endswith.(trainlist, "hdf5")];
    hdf5train = open(joinpath(savedir, "data_train.txt"), "w");
    hdf5validate = open(joinpath(savedir, "data_validate.txt"), "w");
    id = rand(1:length(trainlist), 1)[1];
    for i = 1:length(trainlist)
        i != id ? write(hdf5train, "$(trainlist[i])\n") : write(hdf5validate, "$(trainlist[i])\n");
    end
    close(hdf5train);
    close(hdf5validate);

end




# p = ["helical_extend"; "helical_form_ratio"; "helical_hrise"; "helical_htwist"; "helical_radius"];
# for parm in p
#     helix_collect_data(parm);
# end




# function that save the data to the specified directory
function helix_collect_data_add_const(parameter::String; datsourcedir::String="F:\\Yen\\HJH_KCl_Hydration", dssrsourcedir::String="G:\\My Drive\\13. Julia\\05. WAXSiS\\x3dnadata", targetdir::String="F:\\Yen\\DuplexData_Full", constantdir::String="G:\\My Drive\\13. Julia\\05. WAXSiS\\analysesdata")

    h2_electrons = 4390.54;

    dict = Dict([("helical_extend", :helical_extend), ("helical_form_ratio", :helix_form), ("helical_hrise", :helical_rise), ("helical_htwist", :h_twist), ("helical_radius", :helical_radius)]);

    categories = ["30"; "50"; "200"; "500"];

    savedir = joinpath(targetdir, parameter);

    function clean(x::DSSR, parameter::String; dict::Dict{String, Symbol}=dict)

        fd = dict[parameter];
        tmp = getfield(x, fd);
        data = -1.0;
        if parameter == "helical_extend"
            data = tmp;
        elseif parameter == "helical_form_ratio"
            data = sum(tmp .== 'A') / length(tmp);
        elseif parameter == "helical_hrise"
            data = tmp[1];
        elseif parameter == "helical_htwist"
            data = sum(tmp);
        elseif parameter == "helical_radius"
            data = tmp[1];
        else
            error("Parameter $parameter is not recognized. ");
        end
        return data;
    end

    # Data augmentation by incorportating the constant c for buffer subtraction and dark current
    #augc = [0.0; 1.0e5; 1.5e5; 2.0e5; 2.5e5; 3.0e5];

    # Add oversampling options


    for c in categories

        println("Category $c :");

        # Load the dssr data
        dssrfile = "x3dna$c.jld";
        x3 = load(joinpath(dssrsourcedir, dssrfile), "dssr$c");
        n = length(x3)
        label = zeros(n);
        [label[i] = clean(x3[i], parameter) for i = 1:n];

        # Load the .dat data
        datdir = joinpath(datsourcedir, "KCl$(c)mM");
        datlist = readdir(datdir);
        datlist = datlist[endswith.(datlist, ".dat")];

        # load the constant data
        ana = load(joinpath(constantdir, "h2k$(c).jld"), "h2k$(c)");


        for datfile in datlist

            data = readdlm(joinpath(datdir, datfile));
            data = data[:, 2:end];

            # add the experimentally derived constant
            dc = data[1, :] .* ana.dc ./ ana.nfit[1, 2:end];
            data = data .+ dc';
            data = h2_electrons^2 .* data ./ data[1, :]';

            data = log10.(data);
            @info("datasize = $(size(data, 2)); labelsize = $n. ");

            h5filename = joinpath(savedir, replace(datfile, ".dat" => ".hdf5"));
            h5open(h5filename, "w") do file
                write(file, "data", data, "label", label);
            end
            println("$h5filename written successfully.");
        end
    end

    trainlist = readdir(savedir);
    trainlist = trainlist[endswith.(trainlist, "hdf5")];
    hdf5train = open(joinpath(savedir, "data_train.txt"), "w");
    hdf5validate = open(joinpath(savedir, "data_validate.txt"), "w");
    id = rand(1:length(trainlist), 1)[1];
    for i = 1:length(trainlist)
        i != id ? write(hdf5train, "$(trainlist[i])\n") : write(hdf5validate, "$(trainlist[i])\n");
    end
    close(hdf5train);
    close(hdf5validate);

end


p = ["helical_extend"; "helical_form_ratio"; "helical_hrise"; "helical_htwist"; "helical_radius"];
for parm in p
    helix_collect_data_add_const(parm);
end
