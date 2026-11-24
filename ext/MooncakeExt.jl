module MooncakeExt

import Mooncake
import UnderwaterAcoustics: _mode_cache_get, _mode_cache_put!, _depth_grid, tmap

Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(_mode_cache_get),Any,Any}
Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(_mode_cache_put!),Any,Any,Any}
Mooncake.@from_rrule Mooncake.MinimalCtx Tuple{typeof(_depth_grid),Float64,Float64,Int}

# Mooncake cannot differentiate through threads, so it sees tmap as map
Mooncake.@mooncake_overlay tmap(f, x::AbstractArray) = map(f, x)

end # module
