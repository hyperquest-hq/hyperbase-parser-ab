# Changelog

## [0.4.0] - work in progress

### Added

- Beam search.
- Exact search.
- Structural distortion metric.
- Atom classifier fixed cases.
- Parse trace reports in the repl.
- use_atomizer_subtype option (defaults to True).
- atomizer_model_path, uses local model if set (default is None).

### Changed

- Simplified parses, "r" argrole no longer produced.
- Atomizer provides alternative labels for search.
- Adapted to REPL API (multiple results per parse possible).

### Removed

- Strict rules.
- Parameters: `beta`, `normalise` and `post_process`.

## [0.3.0] - 11-04-2026

### Added

- Maximum depth protection.
- Conjunction flattening.
- Show dependency parse tree in the repl.
- lang_namespace parameter, defaults to False (no language namespaces in atoms).

### Changed

- Adopted new hyperbase API (0.10.0).
- Adopted REPL API.

## [0.2.0] - 05-04-2026

### Changed

- Adopted new hyperbase API (0.9.0).

## [0.1.0] - 02-04-2026 - extracted from graphbrain

### Added

- Atomizier, a multilingual classifier for atom types.
- Can now parse all languages supported by spaCy.

### Changed

- Original alpha-beta parser from Graphbrain was extracted to create this plugin.
