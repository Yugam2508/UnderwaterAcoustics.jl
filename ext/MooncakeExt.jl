module MooncakeExt

import Mooncake
import UnderwaterAcoustics: _mode_cache_get, _mode_cache_put!, _depth_grid

Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(_mode_cache_get),Any,Any}
Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(_mode_cache_put!),Any,Any,Any}
Mooncake.@from_rrule Mooncake.MinimalCtx Tuple{typeof(_depth_grid),Float64,Float64,Int}

end # module
