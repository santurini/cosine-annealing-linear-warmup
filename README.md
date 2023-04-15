# Cosine Annealing Scheduler with Linear Warmup

Implementation of a Cosine Annealing Scheduler with Linear Warmup and Restarts in PyTorch. \
It has support for multiple parameters groups and minimum target learning rates. \
Also works with the Lightning Modules!

# Installation

```pip install 'git+https://github.com/santurini/cosine-annealing-linear-warmup'```

# Usage

It is important to specify the parameters groups in the optimizer instantiation as the learning rates are directly inferred from the wrapped optimizer.

#### Example: Multiple groups

```
from cosine_warmup import CosineAnnealingLinearWarmup

optimizer = torch.optim.Adam([
    {"params": first_group_params, "lr": 1e-3},
    {"params": second_group_params, "lr": 1e-4},
    ]
)

scheduler = CosineAnnealingLinearWarmup(
    optimizer = optimizer,
    min_lrs = [ 1e-5, 1e-6 ],
    first_cycle_steps = 1000,
    warmup_steps = 500,
    gamma = 0.9
    )
    
# this is equivalent to

scheduler = CosineAnnealingLinearWarmup(
    optimizer = optimizer,
    min_lrs_pow = 2,
    first_cycle_steps = 1000,
    warmup_steps = 500,
    gamma = 0.9
    )
```

#### Example: Single groups

```
from cosine_linear_warmup import CosineAnnealingLinearWarmup

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = CosineAnnealingLinearWarmup(
    optimizer = optimizer,
    min_lrs = [ 1e-5 ],
    first_cycle_steps = 1000,
    warmup_steps = 500,
    gamma = 0.9
    )
    
# this is equivalent to

scheduler = CosineAnnealingLinearWarmup(
    optimizer = optimizer,
    min_lrs_pow = 2,
    first_cycle_steps = 1000,
    warmup_steps = 500,
    gamma = 0.9
    )
```

# Visual Example

![Unknown-2](https://user-images.githubusercontent.com/91251307/232208248-a1aa9546-39ff-4456-936a-4953a3cb0d27.png)
