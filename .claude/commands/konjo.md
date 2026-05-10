Run the Konjo session boot sequence for lopi.

This invokes the konjo-boot skill:
1. Read CLAUDE.md, README.md, CHANGELOG.md, PLAN.md, MASTER_PLAN.md, docs/ in order
2. Produce a Session Brief (REPO / LAST SHIPPED / OPEN WORK / BLOCKERS / HEALTH)
3. Run the Discovery protocol — search arXiv, GitHub, HuggingFace for relevant recent developments
4. Identify current work from PLAN.md or propose a sprint if no plan exists

See .claude/skills/konjo-boot/SKILL.md for the full procedure.
