# Release notes

## Parent storage rollout

`parent_storage_mode: classic` stores larger parent chunks in a dedicated
`<collection>_parents` collection and adds `parent_id` to child chunk payloads.
Agora collections are not versioned, and ingestion does not remove stale points
incrementally.

When enabling or disabling parent storage, or when changing
`parent_target_tokens`, `parent_overlap_tokens`, or `parent_max_tokens`, deploy
the ingestion run with `--drop-collection`. This recreates both the main
collection and the parents collection so stale parent points from an older
policy do not remain queryable.
