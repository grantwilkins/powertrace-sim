"""Tool-class taxonomy for the agentic gap model (single source of truth).

Both the offline fitter (``openhands_gap_fit``) and the live sampler
(``gap_sampler``) import ``classify`` + ``IS_REMOTE`` from here so the class
labels and the remote flag can never drift apart. The four classes follow the
empirical split in the research synthesis: local file ops are tight and
near-deterministic, bash is wide (3 orders of magnitude), and remote/subagent
are the heavy tails.
"""

from __future__ import annotations

LOCAL_IO = "local_io"
BASH = "bash"
REMOTE = "remote"
SUBAGENT = "subagent"
CLASSES = (LOCAL_IO, BASH, REMOTE, SUBAGENT)

# Substring -> class. Checked in order; first hit wins. Lower-cased tool name.
_RULES = (
    # subagent / fan-out first (a name like "task_agent" should not match "task" io)
    (("subagent", "sub_agent", "agent", "task", "delegate"), SUBAGENT),
    # remote / network
    (("webfetch", "web_fetch", "websearch", "web_search", "browse", "browser",
      "fetch", "http", "api", "retriev", "search_web", "url", "curl", "wget",
      "request"), REMOTE),
    # shell / command execution
    (("bash", "shell", "execute_bash", "run_command", "cmd", "terminal",
      "exec", "python", "pytest", "pip", "make", "compile", "build"), BASH),
    # local file IO / editing (default-ish bucket for editor tools)
    (("edit", "str_replace", "write", "create", "read", "view", "open",
      "grep", "find", "ls", "cat", "glob", "todo", "file"), LOCAL_IO),
)


def classify(tool_name: str) -> str:
    """Map a raw tool name (e.g. ``str_replace_editor``, ``execute_bash``,
    ``web_fetch``) to one of ``CLASSES``. Unknown names default to ``bash`` —
    the widest class — so an unrecognised tool gets a permissive (high-variance)
    gap rather than a falsely tight ``local_io`` one.
    """
    name = (tool_name or "").strip().lower()
    for needles, cls in _RULES:
        if any(n in name for n in needles):
            return cls
    return BASH
