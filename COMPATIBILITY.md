# Compatibility directories

The public implementation is `setcommit/`. The directories
`dacs_repro/`, `dacs_method_v5/`, and `dacs_method_experiment/` are retained
only because the checked-in torch model runner and historical result scripts
import those module paths. Their frozen policy values map to the public
policies as follows:

```text
dacs_v5_exchange_020 -> setcommit.NATIVE_FAST
dacs_v5_quality_guard -> setcommit.EB_QUALITY_GUARD
```

New integrations should import `setcommit`, not the legacy paths. Keeping the
bridge paths makes old campaign manifests and the reviewer LLaDA runner
reproducible without changing the paper's public API.
