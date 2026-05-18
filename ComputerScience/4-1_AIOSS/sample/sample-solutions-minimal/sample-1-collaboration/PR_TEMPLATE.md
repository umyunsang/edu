# Sample 1 PR Description

## Changes
- Updated `app/greeting.py`.
- Added `get_greeting(name)` behavior for named users.
- Added fallback greeting behavior for empty names.

## Tests
- `python3 tools/aioss_eval/sample_eval.py --target sample/sample-solutions-minimal --label minimal`
- Manual cases: `get_greeting("AIOSS")` and `get_greeting("")`.

## Rollback Plan
- Revert the greeting implementation commit if the behavior causes a regression.

## Checklist
- [x] Functional behavior checked
- [x] Self review completed
- [x] Test evidence recorded
