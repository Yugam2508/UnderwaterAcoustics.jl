module MooncakeExt

import Mooncake
import UnderwaterAcoustics: _mode_cache_get, _mode_cache_put!

Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(_mode_cache_get),Any,Any}
Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(_mode_cache_put!),Any,Any,Any}

end # module
