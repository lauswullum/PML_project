
using Turing
using Turing: Variational
using CSV
using Plots
using ImageView 
using ReverseDiff
Turing.setadbackend(:reversediff)
using LinearAlgebra, FillArrays

# Set a seed for reproducibility.
using Random
Random.seed!(1789);

using MLDatasets

# Load training data (images, labels)

x_train, y_train = MLDatasets.MNIST(split=:train)[:]

# Load test data (images, labels)
#x_test, y_test = MLDatasets.MNIST(split=:train)[:]

# Convert grayscale to float
x_train = Float32.(x_train)

vec_train = reshape(x_train, (28*28, 60000)) 

vec_train1 = vec_train[:, 1:100]


@model function pPCA(X::AbstractMatrix{<:Real}, k::Int)
    # retrieve the dimension of input matrix X.
    N, D = size(X)

    # weights/loadings W
    W ~ filldist(Normal(), D, k)

    # latent variable z
    Z ~ filldist(Normal(), k, N)

    # mean offset
    μ ~ MvNormal(Eye(D))
    mean = W * Z .+ reshape(μ, 784, 1)
    return X ~ arraydist([MvNormal(m, Eye(N)) for m in eachcol(mean')])
end;

# W is 784 x 2
# Z is 2 x 60_000
# μ is 784
# so mean is 784 x 60_000




k = 10 # k is the dimension of the projected space, i.e. the number of principal components/axes of choice
ppca = pPCA(vec_train1', k) # instantiate the probabilistic model

chain_ppca = sample(ppca, NUTS(), 100)

# Extract parameter estimates for predicting x - mean of posterior
W = reshape(mean(group(chain_ppca, :W))[:, 2], (784, k))
Z = reshape(mean(group(chain_ppca, :Z))[:, 2], (k, 100))
μ = mean(group(chain_ppca, :μ))[:, 2]



mat_rec = W * Z .+ repeat(μ; inner=(1, 100))

t = DataFrame(mat_rec, :auto)
CSV.write("post_pred_recon.csv", t)

# plot chain for specific parameter
# Plots.plot(chain_ppca["W[1,1]"])

# Plot a posterior predictive number. 
im1 = reshape(mat_rec[:, 1], 28,28)
ImageView.imshow(im1)




