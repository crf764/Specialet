include("competing_methods.jl")
include("bma.jl")
include("MA2SLS.jl")


function gen_instr_coeff(p, c_M)
    res = zeros(p)
    for i in 1:p
        if i <= p/2
            res[i] = c_M * (1 - i/(p/2 + 1))^4
        end
    end
    return res
end
function givbma_res(y, x, Z, W, y_h, x_h, Z_h, W_h; g_prior = "BRIC", two_comp = false)
    res = givbma(y, x, Z, W; g_prior = g_prior, two_comp = two_comp, iter = 1200, burn = 200)
    lps_int = lps(res, y_h, x_h, Z_h, W_h)
    return (
        τ = mean(rbw(res)[1]),
        CI = quantile(rbw(res)[1], [0.025, 0.975]),
        lps = lps_int
    )
end

function bma_res(y, X, Z, y_h, X_h, Z_h; g_prior = "hyper-g/n")
    res = bma(y, X, Z; g_prior = g_prior, iter = 1200, burn = 200)
    lps_int = lps_bma(res, y_h, X_h, Z_h)
    return (
        τ = mean(rbw_bma(res)[1]),
        CI = quantile(rbw_bma(res)[1], [0.025, 0.975]),
        lps = lps_int
    )
end
function sim_func(m, n; c_M = 3/8, τ = 0.1, p = 20, k = 10, c = 1/2)
    meths = ["BMA (hyper-g/n)", "gIVBMA (BRIC)", "gIVBMA (hyper-g/n)", "gIVBMA (2C)", "IVBMA (KL)", "OLS", "TSLS", "O-TSLS", "JIVE", "RJIVE", "MATSLS", "Post-LASSO"]

    tau_store = Matrix(undef, m, length(meths))
    times_covered = zeros(length(meths))
    lps_store = Matrix(undef, m, length(meths))

    pl_no_instruments = 0 # count how many times post-lasso does not select any instruments

    for i in ProgressBar(1:m)
        d = gen_data_KO2010(n, c_M, τ, p, k, c)
        d_h = gen_data_KO2010(Int(n/5), c_M, τ, p, k, c)

        res = [
            bma_res(d.y, d.x, d.W, d_h.y, d_h.x, d_h.W; g_prior = "hyper-g/n"),
            givbma_res(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.Z, d_h.W; g_prior = "BRIC"),
            givbma_res(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.Z, d_h.W; g_prior = "hyper-g/n"),
            givbma_res(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.Z, d_h.W; g_prior = "hyper-g/n", two_comp = true),
            ivbma_kl(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.Z, d_h.W),
            ols(d.y, d.x, d.W, d_h.y, d_h.x, d_h.W),
            tsls(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W),
            tsls(d.y, d.x, d.Z[:, 1:10], d.W[:, 1:5], d_h.y, d_h.x, d_h.W[:, 1:5]),
            jive(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W),
            rjive(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W),
            matsls(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W),
            post_lasso(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W)
        ]

        tau_store[i,:] = map(x -> x.τ, res)
        lps_store[i, :] = map(x -> x.lps, res)

        # The coverage calculation has to be handled differently if no instruments are selected (just add a zero then)
        if res[end].no_instruments
            pl_no_instruments += 1 # count the number of times post-lasso does not select any instruments
            times_covered += [map(x -> (x.CI[1] < τ < x.CI[2]), res[1:(end-1)]); 0]
        else
            times_covered += map(x -> (x.CI[1] < τ < x.CI[2]), res)
        end
    end

    # We could potentially get an error here if Post-Lasso never selects any instruments, but this should not happen for m large enough
    mae = [median(skipmissing(abs.(tau_store[:, i] .- τ))) for i in eachindex(meths)]
    bias = [(median(skipmissing(tau_store[:, i])) - τ) for i in eachindex(meths)]
    lps = [mean(lps_store[:, 1:(end-1)], dims = 1) missing]
    cov = times_covered ./ [repeat([m], length(meths)-1); m - pl_no_instruments]

    return (MAE = mae, Bias = bias, Coverage = cov, LPS = lps, No_Instruments_PL = pl_no_instruments)


end
function gen_data_KO2010(n = 100, c_M = 3/8, τ = 0.1, p = 20, k = 10, c = 1/2)
    V = rand(MvNormal(zeros(p+k), I), n)'
    V[:, 2:2:end] .*= 100 # adjust scaling by multiplying every other column by 100
    Z = V[:,1:p]
    W = V[:,(p+1):(p+k)]

    α, γ = (1, 1)
    δ_Z = gen_instr_coeff(p, c_M) ./ repeat([1.0, 100.0], Int(p/2))
    δ_W = [ones(Int(k/2)); zeros(Int(k/2))] .* 0.1 ./ repeat([1.0, 100.0], Int(k/2))
    β = [ones(Int(k/2)); zeros(Int(k/2))] ./ repeat([1.0, 100.0], Int(k/2))

    u = rand(MvNormal([0, 0], [1 c; c 1]), n)'
    x = γ .+ Z * δ_Z + W * δ_W + u[:,2]
    y = α .+ τ * x .+ W * β + u[:,1]

    return (y=y, x=x, Z=Z, W=W)
end

