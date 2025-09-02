from __future__ import annotations
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from app.agent.llm import get_llm
from app.agent.tools import TOOLS

SYSTEM = (
    "You are a workout voice assistant that ONLY controls workout rep counting. "
    "Use tools ONLY when the user clearly asks to start/pause/resume/stop/status a workout. "
    "Supported exercises: bicep curl, bench press, lateral raise, shoulder press. "
    "If the user says anything unrelated (e.g., 'record my run', 'open music'), "
    "DO NOT call tools and respond concisely that it's not a workout command."
)

FEWSHOTS = [
    ("start counting my curls", "start_rep_counter"),
    ("pause", "pause_rep_counter"),
    ("resume the set", "resume_rep_counter"),
    ("stop the counter", "stop_rep_counter"),
    ("how many have I done?", "status_rep_counter"),
    ("record my run", "noop"),
]

_TOOL_MAP: Dict[str, Any] = {t.name: t for t in TOOLS}

def _blank_result(user_text: str) -> dict:
    return {
        "transcript": user_text,
        "action": "noop",
        "args": {},
        "tool_output": "",
        "llm_text": "",
        "steps": [],
    }

def route_and_execute(user_text: str, verbose: bool = True) -> dict:
    steps: List[str] = []
    result = _blank_result(user_text)
    steps.append(f"router: received transcript → {user_text!r}")

    # Build messages
    messages = [SystemMessage(content=SYSTEM)]
    for u, label in FEWSHOTS:
        messages.append(HumanMessage(content=u))
        if label == "noop":
            messages.append(SystemMessage(content="Not a workout command. Do not call tools."))
        else:
            messages.append(SystemMessage(content=f"Call tool: {label}"))
    messages.append(HumanMessage(content=user_text))

    # Init LLM safely
    try:
        llm = get_llm().bind_tools(TOOLS)  # bind ALL tools at once
    except Exception as e:
        steps.append(f"router: LLM init error → {e!r}")
        result["llm_text"] = "LLM unavailable (check OPENAI_API_KEY)."
        result["steps"] = steps
        return result

    # Call LLM safely
    try:
        res = llm.invoke(messages)
    except Exception as e:
        steps.append(f"router: LLM call error → {e!r}")
        result["llm_text"] = "LLM call failed (auth/network)."
        result["steps"] = steps
        return result

    # Extract tool calls robustly
    result["llm_text"] = (getattr(res, "content", "") or "").strip()
    tcalls = []
    try:
        # Newer LC puts tool calls on res.tool_calls (list of dicts)
        tcalls = getattr(res, "tool_calls", None)
        if not tcalls:
            # Some versions stash them in additional_kwargs
            ak = getattr(res, "additional_kwargs", None)
            if isinstance(ak, dict):
                tcalls = ak.get("tool_calls") or []
        if tcalls is None:
            tcalls = []
    except Exception as e:
        steps.append(f"router: could not read tool_calls → {e!r}")
        tcalls = []

    if not tcalls:
        steps.append("router: no tool selected (noop)")
        result["steps"] = steps
        return result

    # Execute first tool call
    try:
        tc = tcalls[0]
        name = tc.get("name")
        args = tc.get("args") or {}
        steps.append(f"router: selected tool → {name} with args {args}")
        tool = _TOOL_MAP.get(name)
        if tool is None:
            steps.append(f"router: unknown tool {name} (noop)")
            result["steps"] = steps
            return result
        out = tool.invoke(args)  # our @tool returns a string
        result["action"] = name
        result["args"] = args
        result["tool_output"] = out
        steps.append(f"tool: executed {name} → {out}")
        if name.startswith("start_"): steps.append("camera: starting...")
        if name.startswith("pause_"): steps.append("session: paused")
        if name.startswith("resume_"): steps.append("session: resumed")
        if name.startswith("stop_"): steps.append("session: stopped")
    except Exception as e:
        steps.append(f"tool: ERROR during execution → {e!r}")
        # keep result as noop on error

    result["steps"] = steps
    return result
