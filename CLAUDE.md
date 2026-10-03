@AGENTS.md
@docs/STATUS.md

Claude Code specifics:
- Training runs take hours on the single GPU. Launch them in the background (`nohup ... > runs/<name>.out 2>&1 &`
  or tmux) and poll `runs/<name>/log.jsonl` instead of blocking on them.
- Update `docs/STATUS.md` when a milestone changes (data prepared, a config finished, a bug found),
  with dates and the key numbers, so the next session starts from the truth.
