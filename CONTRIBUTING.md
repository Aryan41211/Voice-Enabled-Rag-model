# Contributing (Team Workflow)

Lightweight, hackathon-scoped — the point is to avoid merge conflicts and lost work in an 8-day sprint, not process for its own sake.


## Commit Convention
`<type>: <short description>` — types: `feat`, `fix`, `docs`, `bench`, `refactor`, `test`.
Example: `feat: add hybrid RRF fusion for dense+BM25 retrieval`

## Daily Sync
- 10-min standup: what shipped yesterday, what's today, any blockers.
- Update `ROADMAP.md` checkboxes as you go — it's your source of truth for what's actually done.

## PR Checklist (even for a hackathon, keep this fast)
- [ ] Runs locally from a clean clone
- [ ] No hardcoded API keys (use `.env`)
- [ ] Touches only your workstream's folder where possible, to minimize conflicts
- [ ] If touching the harness's shared schemas (see `API.md`), ping the team before merging
- [ ] Update `CHANGELOG.md` for anything user-facing or schema-breaking

## Code Ownership (mirrors ROADMAP.md ownership table)
Keep it simple: whoever owns a workstream folder (`app/stt/`, `app/retrieval/`, etc.) is the default reviewer for changes there.
