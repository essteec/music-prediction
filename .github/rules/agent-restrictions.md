---
paths:
  - "**"
---
# Agent Restrictions

## Execution Policy

- **CRITICAL**: Under any circumstances, the agent **MUST NOT RUN** any Python script in the project.
- If a task requires running a Python script, the agent must provide the command and instructions for the **user** to run it instead.
- This rule is **non-negotiable** and applies to all agents, assistants, and automated tools.

## Rationale

This is a completed research project. To ensure the integrity of the results and avoid accidental modifications to the environment or data, all code execution must be performed and verified by the user.
