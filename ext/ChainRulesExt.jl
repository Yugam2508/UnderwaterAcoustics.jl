module ChainRulesExt

using ChainRulesCore
import UnderwaterAcoustics: _arr2ir, tmap, _blackman, _bisect, _mode_cache_get, _mode_cache_put!, _depth_grid
import UnderwaterAcoustics: _nextfastfft, _tukey

# the bisection root-find is value-only; derivatives are recovered by the
# Newton polish step in arrivals (implicit function theorem)
ChainRulesCore.@non_differentiable _bisect(f, lo, hi)
ChainRulesCore.@non_differentiable _mode_cache_get(pm, ω)
ChainRulesCore.@non_differentiable _mode_cache_put!(pm, ω, γ)

ChainRulesCore.@non_differentiable _nextfastfft(n)
ChainRulesCore.@non_differentiable _tukey(n, α)

function ChainRulesCore.rrule(::typeof(_depth_grid), z1, z2, n)
  t = (0:n-1) / (n - 1)
  _depth_grid_pullback(Δ) = (NoTangent(), sum((1 .- t) .* unthunk(Δ)), sum(t .* unthunk(Δ)), NoTangent())
  _depth_grid(z1, z2, n), _depth_grid_pullback
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(tmap), f, X::AbstractArray)
  cache = tmap(X) do x
    rrule_via_ad(config, f, x)
  end
  Y = map(first, cache)
  function tmap_pullback(dY_raw)
    dY = unthunk(dY_raw)
    backevals = map(cache, dY) do (y, back), dy
      back(dy)
    end
    df = ProjectTo(f)(sum(first, backevals))
    dX = map(last, backevals)
    return (NoTangent(), df, dX)
  end
  return Y, tmap_pullback
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(tmap), f, T::Type, X::AbstractArray)
  cache = tmap(X) do x
    rrule_via_ad(config, f, x)
  end
  Y = map(T∘first, cache)
  function tmap_pullback(dY_raw)
    dY = unthunk(dY_raw)
    backevals = map(cache, dY) do (y, back), dy
      back(dy)
    end
    df = ProjectTo(f)(sum(first, backevals))
    dX = map(last, backevals)
    return (NoTangent(), df, NoTangent(), dX)
  end
  return Y, tmap_pullback
end

end # module
