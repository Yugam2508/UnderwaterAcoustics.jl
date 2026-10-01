import SignalAnalysis: duration, nchannels, SampledSignal, samples, signal
import SignalAnalysis: framerate, nframes, resample, isanalytic, analytic, padded
import Interpolations: interpolate, BSpline, Cubic, Line, OnGrid, scale, extrapolate
import Random: AbstractRNG, randexp
import LinearAlgebra: mul!

export BasebandReplayChannel, ReplayNoise

# fs, fc and doppler stay Float64 even when T1 is Float32, since errors in them accumulate over time
struct BasebandReplayChannel{T1,T2} <: AbstractChannelModel
  h::Array{Complex{T1},3}   # channel impulse responses (delay × rx × time)
  θ::Matrix{T1}             # theta_hat phase estimates (time × rx), or 0×0 if unused
  φ::Matrix{T1}             # phi_hat phase estimates (time × rx), or 0×0 if unused
  fs::Float64               # delay-axis sampling rate, fs_delay (Sa/s)
  fc::Float64               # carrier frequency (Hz)
  step::Int                 # step size for h time axis (fs ÷ step IRs/s)
  doppler::Float64          # passband resampling factor (f_resamp)
  noise::T2
  function BasebandReplayChannel(h, θ::AbstractMatrix, φ::AbstractMatrix, fs::Number, fc::Number, step::Int=1, doppler::Real=1.0; noise=nothing)
    fs = in_units(u"Hz", fs)
    fc = in_units(u"Hz", fc)
    T1 = float(real(eltype(h)))
    new{T1,typeof(noise)}(Complex{T1}.(h), T1.(θ), T1.(φ), Float64(fs), Float64(fc), step, Float64(doppler), noise)
  end
end

"""
    Float32(ch::BasebandReplayChannel)
    Float64(ch::BasebandReplayChannel)

Convert the impulse responses and phase estimates of a replay channel to the
given precision. `Float32(ch)` halves the memory used by a channel loaded from
a file.
"""
(::Type{T})(ch::BasebandReplayChannel) where {T<:AbstractFloat} =
  BasebandReplayChannel(Complex{T}.(ch.h), ch.θ, ch.φ, ch.fs, ch.fc, ch.step, ch.doppler; noise=ch.noise)

function Base.show(io::IO, ch::BasebandReplayChannel)
  print(io, "BasebandReplayChannel($(size(ch.h,2)) × $(round(size(ch.h,3)/ch.fs*ch.step; digits=1)) s, $(ch.fc) Hz, $(ch.fs) Sa/s)")
end

"""
    BasebandReplayChannel(h, θ, φ, fs, fc, step=1, doppler=1.0; noise=nothing)
    BasebandReplayChannel(h, θ, fs, fc, step=1; noise=nothing)
    BasebandReplayChannel(h, fs, fc, step=1; noise=nothing)

Construct a baseband replay channel with impulse responses `h` and optional
phase estimates `θ` (`theta_hat`, phase tracking only) or `φ` (`phi_hat`, delay
tracking). `fs` is the sampling frequency in Sa/s, `fc` is the carrier frequency
in Hz, and `step` is the decimation rate for the time axis of `h`. The effective
sampling frequency of the impulse responses is `fs ÷ step` impulse responses per
second. `doppler` is a time-invariant passband resampling factor. The impulse
responses and phase estimates are stored at the precision of `h`.

To use `φ` without `θ`, pass `zeros(0, 0)` for `θ`. If both are given, `φ`
takes precedence.

An additive noise model may be optionally specified as `noise`. If specified,
it is used to corrupt the received signals.
"""
function BasebandReplayChannel(h, θ::AbstractMatrix, fs::Number, fc::Number, step::Int=1; noise=nothing)
  φ = Matrix{Float64}(undef, 0, 0)
  BasebandReplayChannel(h, θ, φ, fs, fc, step; noise)
end

function BasebandReplayChannel(h, fs::Number, fc::Number, step::Int=1; noise=nothing)
  θ = Matrix{Float64}(undef, 0, 0)
  φ = Matrix{Float64}(undef, 0, 0)
  BasebandReplayChannel(h, θ, φ, fs, fc, step; noise)
end

"""
    BasebandReplayChannel(filename; upsample=false, rxs=:, noise=nothing)

Load a baseband replay channel from a file.

If `upsample` is `true`, the impulse responses are upsampled to the delay axis
sampling rate. This makes applying the channel faster but requires more memory.
`rxs` controls which receivers to load from the file. By default, all receivers
are loaded.

An additive noise model may be optionally specified as `noise`. If specified,
it is used to corrupt the received signals. A `ReplayNoise` model is restricted
to the receivers `rxs`, and so must be constructed with all receivers, or with
the same `rxs`.

Supported formats:
- `.mat` (MATLAB) file in underwater acoustic channel repository (UACR) format.
  See https://github.com/uwa-channels/ for details. Loading `.mat` files
  requires the `MAT` package to be loaded (`using MAT`).
"""
function BasebandReplayChannel(filename::AbstractString; upsample=false, rxs=:, noise=nothing)
  endswith(filename, ".mat") || error("Unsupported file format")
  applicable(_load_mat_replay_channel, filename, upsample, rxs, noise) ||
    error("Loading .mat replay channels requires the MAT package; run `using MAT` first")
  _load_mat_replay_channel(filename, upsample, rxs, noise)
end

# implemented in MATExt
function _load_mat_replay_channel end

"""
    transmit(ch::BasebandReplayChannel, x; txs=:, rxs=:, abstime=false, noisy=true, fs=nothing, start=nothing)

Simulate the transmission of passband signal `x` through the channel model `ch`.
If `txs` is specified, it specifies the indices of the sources active in the
simulation. The number of sources must match the number of channels in the
input signal. If `rxs` is specified, it specifies the indices of the
receivers active in the simulation. Returns the received signal at the
specified (or all) receivers.

`fs` specifies the sampling rate of the input signal. The output signal is
sampled at the same rate. If `fs` is not specified but `x` is a `SampledSignal`,
the sampling rate of `x` is used; otherwise an error is raised. If the channel
has a passband resampling factor (`doppler`), the output is also resampled by
that factor to reproduce the nominal Doppler offset.

If `abstime` is `true`, the returned signals begin at the start of transmission.
Otherwise, the result is relative to the earliest arrival time of the signal
at any receiver. If `noisy` is `true` and the channel has a noise model
associated with it, the received signal is corrupted by additive noise.

If `start` is specified, it specifies the starting time index in the replay channel.
If not specified, a random start time is chosen.
"""
function transmit(ch::BasebandReplayChannel, x; txs=:, rxs=:, abstime=false, noisy=true, fs=nothing, start=nothing)
  fs === nothing && x isa SampledSignal && (fs = framerate(x))
  L, M, T = size(ch.h)
  maxtime = ((T - 2) * ch.step - L + 1) / ch.fs
  txs === (:) && (txs = 1)
  rxs === (:) && (rxs = 1:M)
  ndims(rxs) == 0 && (rxs = [rxs])
  nchannels(x) == 1 || error("Replay channel has only one transmitter")
  length(txs) == 1 || error("Replay channel has only one transmitter")
  only(txs) == 1 || error("Replay channel has only one transmitter")
  abstime && error("Replay channels do not support absolute time")
  all(rx ∈ 1:M for rx ∈ rxs) || error("Invalid receiver indices ($rxs ⊄ 1:$M)")
  fs === nothing && error("Sampling rate must be specified")
  fs < 2 * ch.fc && error("Signal sampling rate ($fs Hz) is too low for carrier frequency ($(ch.fc) Hz)")
  input_was_analytic = isanalytic(x)
  x = analytic(signal(samples(x), fs))
  x̄ = samples(resample(x .* cispi.(-2 * ch.fc * (0:nframes(x)-1) ./ fs), ch.fs/fs))
  Treq = ceil(Int, (nframes(x̄) + L - 1) / ch.step) + 1
  Treq < T || error("Signal duration ($(round(duration(x); digits=3)) s) exceeds maximum replayable duration ($(floor(maxtime; digits=3)) s)")
  start = something(start, rand(1:T-Treq))
  1 ≤ start ≤ T - Treq || error("Invalid start index ($start ∉ 1:$(T-Treq))")
  ȳ = similar(x̄, nframes(x̄) + L - 1, length(rxs))
  # only the spline in _interp_ir needs the pad; step == 1 must use the exact window
  pad = ch.step == 1 ? 0 : 2
  lo = max(1, start - pad)
  hi = min(T, start + Treq + pad)
  h = @view ch.h[:,rxs,lo:hi]
  _apply_tvir!(ȳ, x̄, ch.step == 1 ? h : _interp_ir(h, ch.step, nframes(ȳ), (start - lo) * ch.step))
  if size(ch.φ, 2) > 0
    i = (start - 1) * ch.step + 1
    φ_seg = @view(ch.φ[i:i+nframes(ȳ)-1, rxs])
    ȳ .*= cis.(φ_seg)
    t = range(0.0, step=1.0/ch.fs, length=nframes(ȳ))
    for (j, _) ∈ enumerate(rxs)
      drift = φ_seg[:, j] ./(2π * ch.fc)
      itp = extrapolate(scale(interpolate(@view(ȳ[:, j]), BSpline(Cubic(Line(OnGrid())))), t), 0.0)
      ȳ[:, j] .= itp.(t .+ drift)
    end
  elseif size(ch.θ, 2) > 0
    # theta_hat: phase only; the delay drift is already in h
    i = (start - 1) * ch.step + 1
    ȳ .*= cis.(@view(ch.θ[i:i+nframes(ȳ)-1, rxs]))
  end
  y = resample(ȳ, fs/ch.fs; dims=1)
  y .*= cispi.(2 * ch.fc * (0:nframes(y)-1) ./ fs)
  isone(ch.doppler) || (y = resample(y, ch.doppler; dims=1))
  input_was_analytic || (y = real(y) .* √2) # undo analytic()'s 1/√2 scaling
  y = signal(y, fs)
  if noisy && ch.noise !== nothing
    noise = _select_receivers(ch.noise, rxs, M)
    if input_was_analytic
      y .+= analytic(rand(noise, size(y); fs))
    else
      y .+= rand(noise, size(y); fs)
    end
  end
  y
end

function _apply_tvir!(y, x, h)
  L = size(h, 1)
  x = padded(x, L - 1)
  for i ∈ 1:size(y,1)
    y[i,:] .= @views transpose(h[:,:,i]) * x[i-L+1:i]
  end
  y
end

function _interp_ir(h, step, n, offset=0)
  L, M, T = size(h)
  out = similar(h, L, M, n)
  ts = range(0.0, step=float(step), length=T)
  for m ∈ 1:M, l ∈ 1:L
    itp = extrapolate(scale(interpolate(@view(h[l, m, :]), BSpline(Cubic(Line(OnGrid())))), ts), 0.0)
    for i ∈ 1:n
      out[l, m, i] = itp(float(i - 1 + offset))
    end
  end
  out
end

################################################################################
### replay noise model

"""
    ReplayNoise(β, fs, α=2.0; rxs=:, σ=1)
    ReplayNoise(filename; rxs=:, σ=1)

Create an ambient noise model from noise statistics measured at sea, as
distributed in underwater acoustic channel repository (UACR) noise files.

Noise is generated by mixing independent symmetric α-stable innovations `η`,
with scale `1/√2`, across receivers and time lags:

    n[i,t] = σ Σⱼ Σₖ β[i,j,k] η[j,t+k-1]

The mixing coefficients `β[i,j,k]` (receiver × receiver × lag) therefore carry
both the correlation of the noise across receivers and its spectral shaping.
They are measured at a sampling rate of `fs` Sa/s. `α` is the stability index
of the noise, in [0.1, 2]. For `α = 2` the innovations are standard normal, the
noise is Gaussian, and UACR files normalize `β` so that the noise power summed
over the receivers equals the number of receivers. For `α < 2` the noise is
impulsive with infinite variance, and UACR files normalize its pseudo-power the
same way instead. Since `β` does not carry the absolute noise level, `σ` (µPa)
scales the noise, e.g. to set the signal-to-noise ratio.

Since the noise is correlated across receivers, the receivers to generate are
chosen when the model is constructed. `rxs` controls which receivers of `β` are
generated; by default, all of them are. A model passed to
`BasebandReplayChannel(filename; rxs, noise)` is restricted to the receivers of
the channel, and `transmit()` on that channel generates noise for the receivers
it simulates.

Supported formats:
- `.mat` (MATLAB) file in UACR noise format. See
  https://github.com/uwa-channels/ for details. Loading `.mat` files requires
  the `MAT` package to be loaded (`using MAT`).
"""
struct ReplayNoise{T} <: AbstractNoiseModel
  β::Array{T,3}     # mixing coefficients (rx × rx × lag)
  fs::Float64       # rate at which the statistics were measured (Sa/s)
  α::Float64        # stability index (2 = Gaussian, < 2 = impulsive)
  rxs::Vector{Int}  # receivers to generate, indexing the first dimension of β
  function ReplayNoise(β::AbstractArray{<:Real,3}, fs::Number, α::Real=2.0; rxs=:, σ::Number=1)
    fs = in_units(u"Hz", fs)
    σ = in_units(u"µPa", σ)
    fs > 0 || error("Sampling rate must be positive (got $fs)")
    isfinite(σ) || error("Noise scale σ must be finite (got $σ)")
    all(isfinite, β) || error("Mixing coefficients must be finite")
    M = size(β, 1)
    size(β, 2) == M || error("Mixing coefficients must be square in the receiver dimensions")
    # MATLAB's stabrnd refuses α < 0.1 for its risk of overflow; overflow is also caught in _mix_noise
    0.1 ≤ α ≤ 2 || error("Stability index must be in [0.1, 2] (got $α)")
    rxs === (:) && (rxs = 1:M)
    ndims(rxs) == 0 && (rxs = [rxs])
    all(rx ∈ 1:M for rx ∈ rxs) || error("Invalid receiver indices ($rxs ⊄ 1:$M)")
    allunique(rxs) || error("Receiver indices must be unique")
    T = float(eltype(β))
    new{T}(T.(σ .* β), Float64(fs), Float64(α), collect(Int, rxs))
  end
end

function ReplayNoise(filename::AbstractString; rxs=:, σ::Number=1)
  endswith(filename, ".mat") || error("Unsupported file format")
  applicable(_load_mat_replay_noise, filename, rxs, σ) ||
    error("Loading .mat noise models requires the MAT package; run `using MAT` first")
  _load_mat_replay_noise(filename, rxs, σ)
end

# implemented in MATExt
function _load_mat_replay_noise end

# noise models other than ReplayNoise treat their receivers as interchangeable
_select_receivers(noise, rxs, M) = noise

function _select_receivers(noise::ReplayNoise, rxs, M)
  length(noise.rxs) == M ||
    error("Noise model generates $(length(noise.rxs)) receivers, but the channel has $M")
  ReplayNoise(noise.β, noise.fs, noise.α; rxs=noise.rxs[rxs])
end

function Base.show(io::IO, noise::ReplayNoise)
  print(io, "ReplayNoise($(length(noise.rxs)) × $(size(noise.β, 3)) taps, α = $(noise.α), $(noise.fs) Sa/s)")
end

function Base.rand(rng::AbstractRNG, noise::ReplayNoise, nsamples::Integer, nch::Integer; fs)
  nch == length(noise.rxs) ||
    error("Noise model generates $(length(noise.rxs)) receivers ($(noise.rxs)), but $nch requested; " *
          "specify matching rxs when constructing the noise model")
  fs = in_units(u"Hz", fs)
  signal(_mix_noise(rng, noise, nsamples, fs), fs)
end

function Base.rand(rng::AbstractRNG, noise::ReplayNoise, nsamples::Integer; fs)
  x = rand(rng, noise, nsamples, 1; fs)
  signal(dropdims(samples(x); dims=2), framerate(x))
end

function _mix_noise(rng, noise::ReplayNoise{T}, nsamples, fs) where T
  M = size(noise.β, 2)
  nlags = size(noise.β, 3)
  K = ceil(Int, nsamples * noise.fs / fs)
  # the innovations are indexed forward in time (η[j,t+k-1]) to match the reference
  # implementations, whereas the file format specification writes η[j,t-k]; this gives
  # the time reverse of the process the specification describes
  η = noise.α == 2 ? randn(rng, T, K + nlags, M) : _sαsrand(rng, T, K + nlags, M, noise.α)
  β = noise.β[noise.rxs,:,:]  # materialized so that each lag is a strided slice
  w = zeros(T, K, length(noise.rxs))
  for k ∈ 1:nlags
    # accumulating in place matters: the temporary would be as large as the output
    @views mul!(w, η[k:k+K-1,:], transpose(β[:,:,k]), true, true)
  end
  fs == noise.fs || (w = resample(w, fs / noise.fs; dims=1))
  # resampling filters are Float64, so convert back to keep the precision of β
  y = T.(@view w[1:nsamples,:])
  all(isfinite, y) || error("Noise with α = $(noise.α) overflows $T; use a larger α" *
                            (T === Float64 ? "" : " or Float64 coefficients"))
  y
end

# symmetric α-stable variates by the Chambers-Mallows-Stuck method (Chambers,
# Mallows & Stuck, J. Amer. Statist. Assoc. 71(354), 1976), scaled by 1/√2 so
# that α = 2 reduces to the standard normal distribution; computed in Float64
# whatever T, since in Float32 u = -π/2 rounds past the pole and cos(u) < 0
function _sαsrand(rng, ::Type{T}, n, m, α) where T
  x = Array{T}(undef, n, m)
  c = 1 / √2
  for i ∈ eachindex(x)
    u = π * (rand(rng) - 0.5)
    w = randexp(rng)
    x[i] = c * (α == 1 ? tan(u) : sin(α * u) / cos(u)^(1/α) * (cos(u - α * u) / w)^((1-α)/α))
  end
  x
end
