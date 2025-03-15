# CPUOptimizer

A CPU implementation of the Adam optimizer, with pytorch bindings.





## Usage:

### Training loop modifiactions

```py
model = ...
loss_fn = ...

# Close over the optimizer before it's defined. This is legal python, and required.
def pipeline_hook(param):
    optimizer.step_param(param)

optimizer = CPUOptimizer(
    params,
    step_kind="torch_adamw",      # Or "adam" or "adamw"
    lr=4e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.0,
    pipeline_hook=pipeline_hook,  # Or None for a drop-in replacement for Adam without pipelining
)

train_loader = ...

for epoch in range(num_epochs):

    for i, (inputs, labels) in enumerate(train_loader):

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)

        # If you defined a pipeline hook, call this before backward().
        optimizer.begin_step()

        # Call backward() as normal.
        # With a pipeline hook, as grads become available during backward() the optimizer step runs asynchronously on CPU.
        # If you didn't define a pipeline hook, no changes.
        loss.backward()

        # Call step() and zero_grad() as normal.
        # If you're using a pipeline hook, the step() function doesn't do the optimizer step, it just waits until the optimizer step is done.
        # If you aren't, it behaves like a normal optimizer step.
        optimizer.step()

        # Zero grads is unchanged
        optimizer.zero_grad()

        # Everything else is unchanged.
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss/100:.4f}')

```
