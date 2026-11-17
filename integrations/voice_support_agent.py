"""Adapter for oguzhanozfe/voice-support-agent, mirroring what its own eval/run.py does.

Run from a checkout of that repo (it needs its own dependencies, so go through uv):

    PYTHONPATH=/path/to/agent-evals uv run python -m agent_evals run \
      /path/to/agent-evals/integrations/voice_support_scenarios.json \
      --agent integrations.voice_support_agent:agent -k 3

Each run gets a fresh temporary SQLite database. Write tools (update_address, create_ticket,
escalate_to_human) wait for a human "yes"; this adapter approves them, or declines when
VOICE_APPROVE=0. Tool calls are read from the session's message history, so read tools such as
get_customer and search_kb are visible too; its HTTP API only exposes the writes that ran. These are the
calls the model requested: a write the agent's gate rejected or the customer declined still appears.
Usage is not reported: agent.py logs tokens to a file but does not return them.
"""
import json
import os
import tempfile
from pathlib import Path

import openai

import agent as support  # the repo's agent.py
import crm
import knowledge

client = openai.OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
                       api_key=os.environ.get("LLM_API_KEY", "unused"), timeout=120)


def agent(turns):
    with tempfile.TemporaryDirectory() as tmp:
        crm.DB_PATH = Path(tmp) / "eval.db"
        support.LOG_PATH = Path(tmp) / "steps.jsonl"
        crm.init_db()
        knowledge.init()  # embeds the knowledge base into this run's database
        session = support.Session("eval")
        reply = None
        for text in turns:
            reply = support.send(session, text, client)
            while reply.pending:
                reply = support.confirm(session, os.environ.get("VOICE_APPROVE", "1") == "1", client)
    calls = [{"name": c["function"]["name"], "arguments": json.loads(c["function"].get("arguments") or "{}")}
             for m in session.messages if m.get("role") == "assistant" for c in m.get("tool_calls") or []]
    return {"answer": reply.text, "tool_calls": calls}
