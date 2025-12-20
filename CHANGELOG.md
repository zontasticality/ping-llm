# Changelog

## [Unreleased] - 2024-12-19

### Added
- Created `docs/network-training/` directory structure
- Consolidated all network training documentation
- Added comprehensive troubleshooting guides
- Created reference documentation section

### Changed
- **BREAKING**: Renamed `scripts/train_with_wandb_sync.py` → `scripts/train.py`
- **BREAKING**: Renamed `scripts/train/modal_train_with_wandb_sync.py` → `scripts/train/modal_wrapper.py`
- Moved `tokenization.py` → `src/MaxText/input_pipeline/network_tokenization.py`
- Updated `run_modal_training.sh` to use new script paths
- Organized documentation into structured hierarchy

### Removed
- Deleted `src/MaxText/input_pipeline/_network_grain_integration.py` (obsolete)
- Consolidated 10+ root-level markdown files into docs directory
- Removed redundant documentation files

### Fixed
- Fixed import path for tokenization module (no more sys.path hacks)
- Cleaned up module dependencies

## Migration Guide

If you have existing scripts or workflows:

### Update Script Calls
```bash
# OLD:
python scripts/train_with_wandb_sync.py ...
modal run scripts/train/modal_train_with_wandb_sync.py::run ...

# NEW:
python scripts/train.py ...
modal run scripts/train/modal_wrapper.py::run ...
```

### Update Documentation References
- Documentation moved from root to `docs/network-training/`
- See `docs/network-training/README.md` for navigation

### No Code Changes Required
- All internal imports updated automatically
- Config files remain unchanged
- Data format unchanged
