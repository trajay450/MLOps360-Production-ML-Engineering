# Architecture

```mermaid
flowchart LR
    N0["Data validation"] --> N1["Train and evaluate"]
    N1["Train and evaluate"] --> N2["Save model"]
    N2["Save model"] --> N3["Serve inference"]
    N3["Serve inference"] --> N4["Monitor drift"]
```

## Module boundaries

Each stage is deliberately implemented as a compact, inspectable reference pipeline. No external production service is required for the demonstration.
