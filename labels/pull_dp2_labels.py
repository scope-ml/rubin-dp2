#!/usr/bin/env python3
"""Pull the Rubin DP2 labels and periods from Orcus.

Reads the two DP2 groups on Orcus (https://orcusgate.org), "Rubin DP2 Variables" and "Rubin DP2 Transients", and
writes one table per group with one row per object that has at least one classification in the LSST/Rubin Taxonomy.

    python pull_dp2_labels.py --token YOUR_ORCUS_TOKEN
    ORCUS_TOKEN=YOUR_ORCUS_TOKEN python pull_dp2_labels.py --format parquet

Output (default): dp2_variables.csv and dp2_transients.csv, columns
    id               DP2 diaObjectId
    ra, dec          position (degrees)
    class            majority class; each person counts once (their latest classification). Classes tied at the
                     top are all listed, separated by "; "
    votes            how many people chose the class, out of how many classified the object, e.g. "2 of 3"
    classifications  every person's latest classification as author:class:probability, separated by "; "
    period           the set period in days: the "period" field of the object's annotation named "period".
                     Empty unless the class is periodic
    groups           the Orcus DP2 group(s) the object is in

You need an Orcus account that is a member of the group(s); a group you cannot access is skipped with a note.
Only the Python standard library is needed (pyarrow for --format parquet).
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

ORCUS = "https://orcusgate.org"
GROUPS = {"variables": "Rubin DP2 Variables", "transients": "Rubin DP2 Transients"}
TAXONOMY = "LSST/Rubin Taxonomy"
# The set period of each object is the "period" field of its annotation with this origin (one per object).
PERIOD_ORIGIN = "period"
# Branches of the taxonomy whose classes carry a period, plus one periodic class that sits under CV.
PERIODIC_BRANCHES = {"periodic", "pulsator", "binary star"}
PERIODIC_EXTRA = {"Cyclotron emitting magnetic white dwarf"}
COLUMNS = ["id", "ra", "dec", "class", "votes", "classifications", "period", "groups"]


def api(token, path, params=None):
    url = ORCUS + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"Authorization": f"token {token}"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                ans = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504) and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise SystemExit(f"Orcus request failed ({e.code}) for {path}: {e.read().decode()[:200]}")
        if ans.get("status") != "success":
            raise SystemExit(f"Orcus request failed for {path}: {ans.get('message')}")
        return ans["data"]


def periodic_classes(hierarchy):
    """All classes inside the periodic branches of the taxonomy tree, plus PERIODIC_EXTRA."""
    out = set(PERIODIC_EXTRA)

    def walk(node, inside):
        inside = inside or node["class"] in PERIODIC_BRANCHES
        if inside:
            out.add(node["class"])
        for child in node.get("subclasses", []):
            walk(child, inside)

    walk(hierarchy, False)
    return out


def latest(items, key="created_at"):
    return max(items, key=lambda x: x.get(key) or "")


def object_row(src, taxonomy_ids, periodic, min_prob, exclude):
    cls = [c for c in src.get("classifications", [])
           if c.get("taxonomy_id") in taxonomy_ids and not c.get("ml")
           and (c.get("probability") is None or c["probability"] >= min_prob)]
    by_person = {}
    for c in cls:
        by_person.setdefault(c.get("author_name") or str(c.get("author_id")), []).append(c)
    calls = {p: latest(cs) for p, cs in by_person.items()}
    calls = {p: c for p, c in calls.items() if c["classification"] not in exclude}
    if not calls:
        return None
    count = Counter(c["classification"] for c in calls.values())
    top = max(count.values())
    winners = sorted(k for k, v in count.items() if v == top)
    period = ""
    if any(w in periodic for w in winners):
        with_period = [a for a in src.get("annotations", [])
                       if a.get("origin") == PERIOD_ORIGIN and isinstance(a.get("data"), dict)
                       and isinstance(a["data"].get("period"), (int, float))]
        if with_period:
            period = latest(with_period, "modified")["data"]["period"]
    return {
        "id": src["id"], "ra": src["ra"], "dec": src["dec"],
        "class": "; ".join(winners),
        "votes": f"{top} of {len(calls)}",
        "classifications": "; ".join(
            f"{p}:{c['classification']}:{c.get('probability')}" for p, c in sorted(calls.items())),
        "period": period,
        "groups": "",
    }


def group_sources(token, gid):
    out, page = [], 1
    while True:
        data = api(token, "/api/sources", {"group_ids": gid, "numPerPage": 500, "pageNumber": page})
        out += data["sources"]
        if not data["sources"] or len(out) >= data.get("totalMatches", 0):
            return out
        page += 1


def write(rows, path, fmt):
    if fmt == "csv":
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema(
            [(c, pa.float64() if c in ("ra", "dec") else pa.string()) for c in COLUMNS])), path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", default=os.environ.get("ORCUS_TOKEN"),
                    help="Orcus API token (or set ORCUS_TOKEN); create one in your Orcus profile")
    ap.add_argument("--groups", nargs="+", choices=sorted(GROUPS), default=sorted(GROUPS),
                    help="which DP2 groups to pull (default: both)")
    ap.add_argument("--min-prob", type=float, default=0.0,
                    help="ignore classifications with probability below this (default 0)")
    ap.add_argument("--exclude-classes", nargs="*", default=[], help="classes to leave out")
    ap.add_argument("--format", choices=["csv", "parquet"], default="csv")
    ap.add_argument("--outdir", default=".", help="where to write the files (default: current directory)")
    a = ap.parse_args()
    if not a.token:
        sys.exit("No Orcus token: pass --token or set ORCUS_TOKEN.")

    taxonomies = [t for t in api(a.token, "/api/taxonomy") if t["name"] == TAXONOMY]
    if not taxonomies:
        sys.exit(f'The "{TAXONOMY}" is not visible to this account; ask to be added to the Rubin DP2 groups on Orcus.')
    taxonomy_ids = {t["id"] for t in taxonomies}
    periodic = set()
    for t in taxonomies:
        periodic |= periodic_classes(t["hierarchy"])

    mine = {g["name"]: g["id"] for g in api(a.token, "/api/groups").get("user_accessible_groups", [])}
    for key in a.groups:
        name = GROUPS[key]
        if name not in mine:
            print(f'Skipping "{name}": this account is not a member of it on Orcus.')
            continue
        rows = []
        for src in group_sources(a.token, mine[name]):
            row = object_row(src, taxonomy_ids, periodic, a.min_prob, set(a.exclude_classes))
            if row:
                row["groups"] = "; ".join(sorted(g["name"] for g in src.get("groups", []) if g["name"] in GROUPS.values()))
                rows.append(row)
        rows.sort(key=lambda r: r["id"])
        path = os.path.join(a.outdir, f"dp2_{key}.{a.format}")
        write(rows, path, a.format)
        print(f'{name}: {len(rows)} classified objects -> {path}')


if __name__ == "__main__":
    main()
