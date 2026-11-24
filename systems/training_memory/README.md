# Activation checkpointing

How much saved-tensor memory does non-reentrant activation checkpointing save for the same transformer block, inputs, and loss?

`python -m systems.training_memory.training_memory`

Saved-tensor bytes dropped from 108,544 to 12,288 while loss and parameter gradients matched. The autograd save hook measures tensors submitted for saving. Views, repeated saves, and allocator behavior affect how that count relates to peak allocation. Try increasing sequence length while holding model width fixed.
