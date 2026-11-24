import Accessors: @set
export AdiabaticExt

"""
    AdiabaticExt(model, env; dx=0.0, dz=0.0, reciprocal=false, kwargs...)

A 2D adiabatic mode propagation model based on a range-independent modal
propagation `model`. The adiabatic mode model supports range-dependent
bathymetry. Any `kwargs` passed in are transferred to the underlying `model`.

`dx` is the step size in the range direction for integration. `dz` is the mesh
size in the depth direction to compute modes. If `dx` and/or `dz` is set to
zero, it is automatically determined (~10 points per wavelength).

Adiabatic mode models are usually not reciprocal, i.e., exchanging a source and
receiver changes the answer. This is because the bathymetry is assumed to have
azimuthal symmetry around the source. In `reciprocal` mode, the adiabatic
computation is modified to ensure reciprocity, but azimuthal symmetry is lost.

If the underlying `model` is differentiable, `AdiabaticExt` supports
forward-mode (e.g. ForwardDiff) and reverse-mode (e.g. Zygote, Mooncake,
Enzyme) automatic differentiation.
"""
struct AdiabaticExt{T1,T2,T3} <: AbstractModePropagationModel
  env::T1
  kwargs::T2
  dx::Float64
  dz::Float64
  reciprocal::Bool
  function AdiabaticExt(model::Type{<:AbstractModePropagationModel}, env; dx=0.0, dz=0.0, reciprocal=false, kwargs...)
    has_scatterers(env) && error("AdiabaticExt does not support scatterers")
    new{typeof(env),typeof(kwargs),model}(env, kwargs, dx, dz, reciprocal)
  end
end

function Base.show(io::IO, pm::AdiabaticExt)
  print(io, "AdiabaticExt($(_model(pm)))")
end

## interface methods

function arrivals(pm::AdiabaticExt, tx::AbstractAcousticSource, rx::AbstractAcousticReceiver)
  min_depth = minimum(pm.env.bathymetry)
  max_depth = maximum(pm.env.bathymetry)
  if min_depth == max_depth
    env = let env = pm.env
      @set env.bathymetry = min_depth
    end
    pm1 = _model(pm)(env; pm.kwargs...)
    return arrivals(pm1, tx, rx)
  end
  # only 2D models in the x-z plane supported at present
  location(tx).y == 0 || error("Source must be in the x-z plane")
  location(rx).y == 0 || error("All receivers must be in the x-z plane")
  # dz is allowable error in z for cached mode, dx is step size in x for integration
  dz = pm.dz > 0 ? pm.dz : minimum(pm.env.soundspeed) / (frequency(tx) * 10)
  dx = pm.dx > 0 ? pm.dx : minimum(pm.env.soundspeed) / (frequency(tx) * 10)
  # precompute modes and cache them
  tx_depth = value(pm.env.bathymetry, location(tx))
  cache = _precompute_arrivals(pm, tx, tx_depth, rx, min_depth, max_depth, dz)
  # compute all modes
  xs = range(location(tx).x, location(rx).x; step = location(tx).x ≤ location(rx).x ? dx : -dx)
  modes = map(x -> _cached_arrivals(cache, -value(pm.env.bathymetry, (x=x, y=0.0, z=0.0)), dz), xs)
  rx_modes = _cached_arrivals(cache, -value(pm.env.bathymetry, (x=location(rx).x, y=0.0, z=0.0)), dz)
  nmodes = min(mapreduce(length, min, modes), length(rx_modes))
  ω = 2π * frequency(tx)
  rv = map(1:nmodes) do m
    if any(map(mode -> mode[m] === nothing, modes))
      ModeArrival(0, rx_modes[m].kᵣ, rx_modes[m].ψ, rx_modes[m].v, rx_modes[m].vₚ)
    else
      v = length(modes) / sum(mode -> 1 / mode[m].v, modes)
      ModeArrival(m, rx_modes[m].kᵣ, rx_modes[m].ψ, v, ω / real(rx_modes[m].kᵣ))
    end
  end
  filter(mode -> mode.m != 0, rv)
end

function acoustic_field(pm::AdiabaticExt, tx::AbstractAcousticSource, rx::AbstractAcousticReceiver; mode=:coherent)
  only(acoustic_field(pm, tx, [rx]; mode))
end

function acoustic_field(pm::AdiabaticExt, tx::AbstractAcousticSource, rxs::AbstractArray{<:AbstractAcousticReceiver}; mode=:coherent)
  min_depth = minimum(pm.env.bathymetry)
  max_depth = maximum(pm.env.bathymetry)
  if min_depth == max_depth
    env = let env = pm.env
      @set env.bathymetry = min_depth
    end
    pm1 = _model(pm)(env; pm.kwargs...)
    return acoustic_field(pm1, tx, rxs; mode)
  end
  # only 2D models in the x-z plane supported at present
  location(tx).y == 0 || error("Source must be in the x-z plane")
  all(map(rx -> location(rx).y == 0, rxs)) || error("All receivers must be in the x-z plane")
  # dz is allowable error in z for cached mode, dx is step size in x for integration
  dz = pm.dz > 0 ? pm.dz : minimum(pm.env.soundspeed) / (frequency(tx) * 10)
  dx = pm.dx > 0 ? pm.dx : minimum(pm.env.soundspeed) / (frequency(tx) * 10)
  # precompute modes and cache them using a virtual rx as far as possible just
  # to help with choice of solver parameters for some models
  tx_depth = value(pm.env.bathymetry, location(tx))
  max_r = maximum(rx -> abs(location(rx).x - location(tx).x), rxs)
  virtual_rx = AcousticReceiver(location(tx).x + max_r, 0, tx_depth)
  cache = _precompute_arrivals(pm, tx, tx_depth, virtual_rx, min_depth, max_depth, dz)
  # get x positions of interest
  xs = unique(sort(map(rx -> location(rx).x, vec(rxs))))
  ix = map(rx -> searchsortedfirst(xs, location(rx).x), rxs)
  x0 = location(tx).x
  z0 = -value(pm.env.bathymetry, (x=x0, y=0.0, z=0.0))
  i0 = something(findfirst(≥(x0), xs), length(xs)+1)
  tx_modes = _cached_arrivals(cache, z0, dz)
  m₁ = tx_modes[findfirst(!isnothing, tx_modes)]
  T = complex(promote_type(typeof(float(x0)), typeof(real(m₁.kᵣ)), typeof(real(m₁.ψ(location(tx).z)))))
  zs = map(x -> -value(pm.env.bathymetry, (x=x, y=0.0, z=0.0)), xs)
  rx_modes = map(z -> _cached_arrivals(cache, z, dz), zs)
  fld = zeros(T, size(rxs))
  for m ∈ eachindex(tx_modes)
    kri = vcat(
      reverse(_kr_integrals(cache, pm.env, m, x0, xs[i0-1:-1:1], dx, dz, T)),
      _kr_integrals(cache, pm.env, m, x0, xs[i0:end], dx, dz, T))
    A = map(eachindex(xs)) do i
      (xs[i] == x0 || kri[i] == 0 || m > length(rx_modes[i])) && return zero(T)
      tx_modes[m].ψ(location(tx).z) * cis(-kri[i]) /
        _csqrt(pm.reciprocal ? kri[i] : rx_modes[i][m].kᵣ * abs(xs[i] - x0))
    end
    fld = fld .+ map(rxs, ix) do rx, i
      xs[i] == x0 && return T(NaN)
      zᵣ = location(rx).z
      (A[i] == 0 || zᵣ < zs[i]) && return zero(T)
      p = A[i] * rx_modes[i][m].ψ(zᵣ)
      T(mode === :coherent ? p : abs2(p))
    end
  end
  mode === :coherent || (fld = _csqrt.(fld))
  fld = fld .* (sqrt(2π) * cispi(-0.25) * db2amp(spl(tx)))
  # conjugation required because we use the convention of cis(-kᵣR) to match with Kraken
  conj.(fld)
end

## private methods

_model(pm::AdiabaticExt{T1,T2,T3}) where {T1,T2,T3} = T3

struct ModeCache{Tz,T}
  zs::Vector{Tz}
  modes::Vector{T}
end

function _cached_arrivals(cache, z, dz)
  i = searchsortedfirst(cache.zs, z)
  i ≤ length(cache.zs) && cache.zs[i] == z && return cache.modes[i]
  dz1 = i > 1 ? z - cache.zs[i-1] : Inf
  dz2 = i ≤ length(cache.zs) ? cache.zs[i] - z : Inf
  if dz1 < dz2
    dz1 > 1.01dz && error("Cache miss for required mode")
    cache.modes[i-1]
  else
    dz2 > 1.01dz && error("Cache miss for required mode")
    cache.modes[i]
  end
end

# trapezoid ∫kᵣdx of mode m from a to b, and whether the mode exists all the way
function _kr_integral(cache, env, m, a, b, dx, dz, T)
  right = a ≤ b
  kri = zero(T)
  x = a
  m1 = _cached_arrivals(cache, -value(env.bathymetry, (x=x, y=0.0, z=0.0)), dz)
  while right ? x < b - dx : x > b + dx
    x = right ? x + dx : x - dx
    m2 = _cached_arrivals(cache, -value(env.bathymetry, (x=x, y=0.0, z=0.0)), dz)
    min(length(m1), length(m2)) < m && break
    (m1[m] === nothing || m2[m] === nothing) && break
    kri += 0.5 * (m1[m].kᵣ + m2[m].kᵣ) * dx
    m1 = m2
  end
  if right ? x < b : x > b
    m2 = _cached_arrivals(cache, -value(env.bathymetry, (x=b, y=0.0, z=0.0)), dz)
    ((right ? x < b - dx : x > b + dx) || min(length(m1), length(m2)) < m) && return kri, false
    kri += 0.5 * (m1[m].kᵣ + m2[m].kᵣ) * (right ? b - x : x - b)
  end
  kri, true
end

function _kr_integrals(cache, env, m, x0, xs, dx, dz, T)
  isempty(xs) && return zeros(T, 0)
  seg = map(i -> _kr_integral(cache, env, m, i == 1 ? x0 : xs[i-1], xs[i], dx, dz, T), eachindex(xs))
  n = something(findfirst(s -> !last(s), seg), length(seg) + 1) - 1
  cumsum(map(first, seg)) .* (eachindex(seg) .≤ n)
end

function _precompute_arrivals(pm, tx, tx_depth, rx, min_depth, max_depth, dz)
  n = ceil(Int, (max_depth - min_depth) / 2dz) + 1
  zs = _depth_grid(-max_depth, -min_depth, n)
  i = searchsortedfirst(zs, -tx_depth)
  zs = i ≤ n && zs[i] == -tx_depth ? zs : vcat(zs[1:i-1], -tx_depth, zs[i:n])
  ModeCache(zs, map(z -> _compute_arrivals(pm, tx, z, rx), zs))
end

_depth_grid(z1, z2, n) = collect(range(z1, z2; length=n))

function _compute_arrivals(pm, tx, z, rx)
  # compute arrivals for a water depth of z
  env = let env = pm.env
    @set env.bathymetry = -z
  end
  pm1 = _model(pm)(env; pm.kwargs...)
  # identity has an rrule, which lets Zygote pass zero tangents for unused modes
  identity(arrivals(pm1, tx, rx))
end

function _align_arrivals!(arr1, arr0; candidates=3, corr_threshold=0.5, npts=101)
  arr2 = similar(arr0)
  arr2 .= nothing
  for m1 ∈ arr1
    zs = range(extrema(m1.ψ.zrange)...; length=npts)
    m1ψ = m1.ψ.(zs)
    ndx = sortperm(map(zip(arr0, arr2)) do (m0, m2)
      m0 === nothing || m2 !== nothing ? Inf : abs(m0.kᵣ - m1.kᵣ)
    end)
    resize!(ndx, min(candidates, length(ndx)))
    v, i = findmax(map(m0 -> abs(m1ψ' * m0.ψ.(zs)), arr0[ndx]))
    v / abs(m1ψ' * m1ψ) > corr_threshold && (arr2[ndx[i]] = m1)
  end
  arr2
end
