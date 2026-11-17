"""Adapter for oguzhanozfe/customer-360-payments-lab (its lab.ask tool-calling loop over read-only SQL).

Run from a checkout of that repo, after `python3 scripts/run_analysis.py` has built data/lab.sqlite:

    PYTHONPATH=/path/to/agent-evals LLM_MODEL=qwen3.5:9b \
      python3 -m agent_evals run /path/to/agent-evals/integrations/customer_360_scenarios.json \
      --agent integrations.customer_360:agent -k 3

lab.ask returns only the run_sql queries, so describe_schema calls are not visible to the checks.
It answers one question per conversation: only the last turn is sent.
"""
from lab.ask import ask, connect


def agent(turns):
    answer, queries = ask(turns[-1], connect(), log=lambda *_: None)
    return {"answer": answer, "tool_calls": [{"name": "run_sql", "arguments": {"query": q}} for q in queries]}
