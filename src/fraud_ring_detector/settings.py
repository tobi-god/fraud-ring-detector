"""Kedro project settings."""
# Soft-merge parameter files so an environment (e.g. `kedro run --env mendeley`) overrides single keys instead of
# replacing whole blocks such as `features:`.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "merge_strategy": {"parameters": "soft"},
}
