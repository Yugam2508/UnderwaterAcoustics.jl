import SignalAnalysis: duration, nchannels, SampledSignal, samples, signal
import SignalAnalysis: framerate, nframes, resample, isanalytic, analytic, padded
import Interpolations: interpolate, BSpline, Cubic, Line, OnGrid, scale, extrapolate

export BasebandReplayChannel

# Phase fields and fc are Float64 regardless of T1: the drift correction
# (φ/2πfc ~ 1e-5 s) and the carrier phasor over long signals both lose
# precision in Float32. h stays ComplexF32 to keep large channels in memory.
struct BasebandReplayChannel{T1,T2} <: AbstractChannelModel
  h::Array{Complex{T1},3}
  θ::Matrix{Float64}        
  φ::Matrix{Float64}       
  fs::T1
  fc::Float64             
  step::Int
  f_resamp::Float64
  noise::T2
  function BasebandReplayChannel(h, θ::AbstractMatrix, φ::AbstractMatrix, fs::Real, fc::Real, step::Int=1, f_resamp::Real=1.0; noise=nothing)
    h = ComplexF32.(h)
    θ = Float64.(θ)
    φ = Float64.(φ)
    new{Float64,typeof(noise)}(h, θ, φ, Float32(fs), Float64(fc), step, Float64(f_resamp), noise)
  end
end

function Base.show(io::IO, ch::BasebandReplayChannel)
  print(io, "BasebandReplayChannel($(size(ch.h,2)) × $(round(size(ch.h,3)/ch.fs*ch.step; digits=1)) s, $(ch.fc) Hz, $(ch.fs) Sa/s)")
end

"""
    BasebandReplayChannel(h, θ, φ, fs, fc, step=1, f_resamp=1.0; noise=nothing)
    BasebandReplayChannel(h, θ, fs, fc, step=1; noise=nothing)
    BasebandReplayChannel(h, fs, fc, step=1; noise=nothing)

Construct a baseband replay channel with impulse responses `h` and optional
phase estimates `θ` (theta_hat, phase tracking only) or `φ` (phi_hat, delay
tracking). `fs` is the sampling frequency in Sa/s, `fc` is the carrier frequency
in Hz, and `step` is the decimation rate for the time axis of `h`. The effective
sampling frequency of the impulse responses is `fs ÷ step` impulse responses per
second. `f_resamp` is a time-invariant passband resampling factor.

Channels are normally loaded from a UACR file (see below), which populates `θ`,
`φ` and `f_resamp` from the file. The constructors above are mainly useful for
synthetic channels: pass an empty `Matrix{Float64}(undef, 0, 0)` for whichever
of `θ` or `φ` is not used. If both are given, `φ` takes precedence.

An additive noise model may be optionally specified as `noise`. If specified,
it is used to corrupt the received signals.
"""
function BasebandReplayChannel(h, θ::AbstractMatrix, fs::Real, fc::Real, step::Int=1; noise=nothing)
  fs = in_units(u"Hz", fs)
  fc = in_units(u"Hz", fc)
  φ = Matrix{Float64}(undef, 0, 0)
  BasebandReplayChannel(h, θ, φ, fs, fc, step; noise)
end

function BasebandReplayChannel(h, fs::Real, fc::Real, step::Int=1; noise=nothing)
  fs = in_units(u"Hz", fs)
  fc = in_units(u"Hz", fc)
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
it is used to corrupt the received signals.

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
specifies a passband resampling factor (`f_resamp`), the output is resampled by that factor to
reproduce the nominal Doppler offset.

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
  maxtime = (T - 1) / ch.fs * ch.step
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
  duration(x) ≤ maxtime || error("Signal duration ($(round(duration(x); digits=1)) s) exceeds replay channel duration ($(round(maxtime; digits=1)) s)")
  # convert to baseband and downsample
  x̄ = samples(resample(x .* cispi.(-2 * ch.fc * (0:nframes(x)-1) ./ fs), ch.fs/fs))
  # choose a random start time if not specified
  Treq = ceil(Int, (nframes(x̄) + L - 1) / ch.step) + 1
  start = something(start, rand(1:T-Treq))
  # apply the channel
  ȳ = similar(x̄, nframes(x̄) + L - 1, length(rxs))
  h = @view ch.h[:,rxs,start:start+Treq]
  _apply_tvir!(ȳ, x̄, ch.step == 1 ? h : _interp_ir(h, ch.step, nframes(ȳ)))
  if size(ch.φ, 2) > 0
    # phi_hat: apply phase then re-interpolate at time-shifted grid to insert delay drift
    i = (start - 1) * ch.step + 1
    φ_seg = @view(ch.φ[i:i+nframes(ȳ)-1, rxs])
    ȳ .*= cis.(φ_seg)
    t = range(0.0, step=1.0/ch.fs, length=nframes(ȳ))
    for (j, _) ∈ enumerate(rxs)
      drift = Float64.(φ_seg[:, j] ./(2π * ch.fc))
      itp = extrapolate(scale(interpolate(@view(ȳ[:, j]), BSpline(Cubic(Line(OnGrid())))), t), 0.0)
      ȳ[:, j] .= itp.(t .+ drift)
    end
  elseif size(ch.θ, 2) > 0
    # theta_hat: phase only, no delay interpolation
    i = (start - 1) * ch.step + 1
    ȳ .*= cis.(@view(ch.θ[i:i+nframes(ȳ)-1, rxs]))
  end
  # resample to original sampling rate and upconvert to passband
  y = resample(ȳ, fs/ch.fs; dims=1)
  y .*= cispi.(2 * ch.fc * (0:nframes(y)-1) ./ fs)
  # resample in passband to reproduce the nominal Doppler offset, if needed
  isone(ch.f_resamp) || (y = resample(y, Float64(ch.f_resamp); dims=1))
  input_was_analytic || (y = real(y) .* √2) # SignalAnalysis.analytic() is energy-preserving (divides by √2)
  y = signal(y, fs)
  # add noise
  if noisy && ch.noise !== nothing
    if input_was_analytic
      y .+= analytic(rand(ch.noise, size(y); fs))
    else
      y .+= rand(ch.noise, size(y); fs)
    end
  end
  y
end

# helpers

function _apply_tvir!(y, x, h)
  L = size(h, 1)
  x = padded(x, L - 1)
  for i ∈ 1:size(y,1)
    y[i,:] .= @views transpose(h[:,:,i]) * x[i-L+1:i]
  end
  y
end

# Interpolate the impulse response along its time axis from the snapshot rate
# (fs_delay/step) up to the delay rate, using a cubic spline with zero fill
# outside the sampled range.
function _interp_ir(h, step, n)
  L, M, T = size(h)
  out = similar(h, L, M, n)
  ts = range(0.0, step=float(step), length=T)     # snapshot times, in delay samples
  for m ∈ 1:M, l ∈ 1:L
    itp = extrapolate(scale(interpolate(@view(h[l, m, :]), BSpline(Cubic(Line(OnGrid())))), ts), 0.0)
    for i ∈ 1:n
      out[l, m, i] = itp(float(i - 1))
    end
  end
  out
end