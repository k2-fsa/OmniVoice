# Release Readiness

## Current Status

PR 184 is ready for maintainer review when all local gates pass and the worktree
is clean. This branch does not claim upstream release authority.

## Ready Criteria

- Worktree clean.
- Branch pushed to the PR head.
- Native tests pass.
- Compile gate passes.
- Secret scan passes.
- Offline provider boundary checks pass.
- FFmpeg smoke passes when FFmpeg is installed.
- SpecOps validate and eval pass.

## Not Included

- Upstream merge decision.
- Live OpenRouter smoke without approval.
- Private manuscript production validation.

## Acceptance Criteria

- Worktree is clean after generated local reports are removed.
- PR body contains current validation evidence.
- External dependencies and unrun live checks are listed honestly.
