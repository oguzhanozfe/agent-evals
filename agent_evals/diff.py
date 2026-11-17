"""Compare two run files scenario by scenario.

A regression is a scenario that got worse: pass -> flaky, pass -> fail, or flaky -> fail.
Scenarios that only exist in one file, and scenarios whose runs all errored, are listed but never
count as regressions: they say nothing about the agent's behavior.
"""
RANK = {"fail": 0, "flaky": 1, "pass": 2}


def diff(old, new):
    a, b = old["scenarios"], new["scenarios"]
    out = {"regressions": [], "improvements": [], "flaky_new": [], "flaky_fixed": [],
           "added": sorted(set(b) - set(a)), "removed": sorted(set(a) - set(b)), "errored": [], "warnings": []}
    if old["meta"].get("k") != new["meta"].get("k"):
        out["warnings"].append(f"k differs: {old['meta'].get('k')} vs {new['meta'].get('k')}")
    for sid in sorted(set(a) & set(b)):
        before, after = a[sid]["status"], b[sid]["status"]
        line = f"{sid}: {before} ({a[sid]['passes']}/{len(a[sid]['runs'])}) -> {after} ({b[sid]['passes']}/{len(b[sid]['runs'])})"
        if "error" in (before, after):
            if before != after:
                out["errored"].append(line)
            continue
        if RANK[after] < RANK[before]:
            out["regressions"].append(line)
        elif RANK[after] > RANK[before]:
            out["improvements"].append(line)
        if after == "flaky" and before != "flaky":
            out["flaky_new"].append(sid)
        if before == "flaky" and after != "flaky":
            out["flaky_fixed"].append(sid)
    return out


def render(d):
    titles = [("regressions", "Newly failing (regressions)"), ("improvements", "Newly passing"),
              ("flaky_new", "Became flaky"), ("flaky_fixed", "No longer flaky"), ("added", "Added scenarios"),
              ("removed", "Removed scenarios"), ("errored", "Adapter errors (not counted)"), ("warnings", "Warnings")]
    lines = []
    for key, title in titles:
        if d[key]:
            lines.append(f"{title}:")
            lines += [f"  {item}" for item in d[key]]
    lines.append(f"{len(d['regressions'])} regression(s)")
    return "\n".join(lines)
