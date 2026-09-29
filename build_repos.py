#!/usr/bin/env python3
"""
build_repos.py - generate small git repos with KNOWN performance regressions,
to check that Culprit finds the right commit, diagnoses it, and fixes it.

Every repo:
  * is pure Python, stdlib only (requirements.txt is comment-only, so Culprit's
    LocalGitSandbox builds one cached venv and never touches PyPI)
  * has `bench.py` (unchanged across the whole history) whose LAST stdout line
    is the score in ms - lower is better - matching Culprit's benchmark contract
  * has deterministic commit dates/authors, so SHAs are identical on every machine
  * is self-validated: the benchmark is run at EVERY commit and the timeline is
    checked (flat before the guilty commit, clearly slower from it onward)

Usage:
    python build_repos.py                 # build + validate all repos into ./repos
    python build_repos.py --only lost     # only repos whose name contains "lost"
    python build_repos.py --fast          # 1 bench run per commit instead of 3
    python build_repos.py --no-validate   # skip the benchmark timeline check
    python build_repos.py --force         # overwrite existing repo dirs
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Union

AUTHOR_NAME = "Culprit Test"
AUTHOR_EMAIL = "culprit-test@example.com"
T0 = datetime(2026, 1, 5, 10, 0, 0, tzinfo=timezone.utc)
BENCH_CMD = "python bench.py"
THRESHOLD_PCT = 15.0  # Culprit's default REGRESSION_THRESHOLD_PCT

Transform = Union[str, Callable[[str], str]]


# --------------------------------------------------------------------------
# tiny DSL
# --------------------------------------------------------------------------

def d(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


def replace(old: str, new: str) -> Callable[[str], str]:
    def f(text: str) -> str:
        assert old in text, f"authoring error: snippet not found:\n{old}"
        return text.replace(old, new, 1)
    return f


def append(block: str) -> Callable[[str], str]:
    def f(text: str) -> str:
        return text.rstrip("\n") + "\n\n\n" + d(block)
    return f


@dataclass
class Commit:
    message: str
    files: dict[str, Transform]


@dataclass
class Repo:
    name: str
    description: str
    expected_category: str | None          # None => no regression expected
    acceptable_categories: list[str]
    guilty: int | None                     # index into commits (0 = baseline)
    fixable: bool
    commits: list[Commit] = field(default_factory=list)
    min_ratio: float = 1.5                 # guilty must be >= this x baseline


def C(message: str, **files: Transform) -> Commit:
    # kwargs can't contain "/" or "." so files are passed via dict(**{...}) below
    return Commit(message, files)


def CF(message: str, files: dict[str, Transform]) -> Commit:
    return Commit(message, files)


def common_files(pkg: str, title: str, blurb: str) -> dict[str, Transform]:
    return {
        "README.md": d(f"""
            # {title}

            {blurb}

            Benchmark: `{BENCH_CMD}` (prints elapsed ms on the last line).

            This repo is a synthetic fixture for testing Culprit. Not real software.
            """),
        "requirements.txt": "# no third-party dependencies\n",
        ".gitignore": "__pycache__/\n*.pyc\n.venv/\n",
        f"{pkg}/__init__.py": '__version__ = "0.1.0"\n',
    }


def bump(pkg: str, old: str, new: str) -> Commit:
    return CF(f"Bump version to {new}", {
        f"{pkg}/__init__.py": replace(f'__version__ = "{old}"', f'__version__ = "{new}"'),
    })


def readme_note(message: str, note: str) -> Commit:
    return CF(message, {"README.md": append(note)})


def helper(message: str, path: str, code: str) -> Commit:
    return CF(message, {path: append(code)})


# --------------------------------------------------------------------------
# Repo A - N+1 queries (guilty: middle)
# --------------------------------------------------------------------------

def repo_n_plus_one() -> Repo:
    good = d('''
        def order_report(conn):
            """Return [(order_id, customer_name, tier, total)] for every order."""
            return conn.execute(
                "SELECT o.id, c.name, c.tier, o.total "
                "FROM orders o JOIN customers c ON c.id = o.customer_id "
                "ORDER BY o.id"
            ).fetchall()
    ''')
    bad = d('''
        def order_report(conn):
            """Return [(order_id, customer_name, tier, total)] for every order."""
            orders = conn.execute(
                "SELECT id, customer_id, total FROM orders ORDER BY id"
            ).fetchall()
            report = []
            for order_id, customer_id, total in orders:
                name, tier = _load_customer(conn, customer_id)
                report.append((order_id, name, tier, total))
            return report


        def _load_customer(conn, customer_id):
            return conn.execute(
                "SELECT name, tier FROM customers WHERE id = ?", (customer_id,)
            ).fetchone()
    ''')
    db_py = d('''
        """Tiny in-memory sqlite dataset for the order report."""
        import random
        import sqlite3


        def make_db(n_customers=500, n_orders=20000, seed=7):
            rnd = random.Random(seed)
            conn = sqlite3.connect(":memory:")
            conn.execute(
                "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, tier TEXT)"
            )
            conn.execute(
                "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL)"
            )
            tiers = ["free", "pro", "team", "enterprise"]
            conn.executemany(
                "INSERT INTO customers VALUES (?, ?, ?)",
                [(i, f"customer-{i}", tiers[i % 4]) for i in range(1, n_customers + 1)],
            )
            conn.executemany(
                "INSERT INTO orders VALUES (?, ?, ?)",
                [
                    (i, rnd.randint(1, n_customers), round(rnd.uniform(5, 500), 2))
                    for i in range(1, n_orders + 1)
                ],
            )
            conn.commit()
            return conn
    ''')
    bench = d('''
        import time

        from shop.db import make_db
        from shop.reports import order_report


        def main():
            conn = make_db()
            t0 = time.perf_counter()
            rows = order_report(conn)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            assert len(rows) == 20000
            print(f"rows: {len(rows)}")
            print(f"{elapsed_ms:.2f}")


        if __name__ == "__main__":
            main()
    ''')
    first = common_files("shop", "shop-orders", "Order report over an in-memory sqlite dataset.")
    first.update({"shop/db.py": db_py, "shop/reports.py": '"""Order reporting."""\n\n\n' + good, "bench.py": bench})

    commits = [
        CF("Initial commit: order report over sqlite", first),
        helper("Add orders_per_tier helper", "shop/reports.py", '''
            def orders_per_tier(conn):
                """Return {tier: order_count}."""
                rows = conn.execute(
                    "SELECT c.tier, COUNT(*) FROM orders o "
                    "JOIN customers c ON c.id = o.customer_id GROUP BY c.tier"
                ).fetchall()
                return dict(rows)
        '''),
        CF("Document make_db parameters", {"shop/db.py": replace(
            '"""Tiny in-memory sqlite dataset for the order report."""',
            '"""Tiny in-memory sqlite dataset for the order report.\n\nmake_db() is deterministic for a given seed."""')}),
        helper("Add customer_count helper", "shop/reports.py", '''
            def customer_count(conn):
                return conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        '''),
        bump("shop", "0.1.0", "0.2.0"),
        readme_note("Add usage section to README", "## Usage\n\n```python\nfrom shop.db import make_db\nfrom shop.reports import order_report\n\nrows = order_report(make_db())\n```\n"),
        CF("Refactor order_report into smaller steps for readability", {"shop/reports.py": replace(good, bad)}),
        helper("Add CSV export helper for reports", "shop/reports.py", '''
            def to_csv(rows):
                lines = ["order_id,customer,tier,total"]
                lines += [",".join(str(c) for c in row) for row in rows]
                return "\\n".join(lines)
        '''),
        readme_note("Mention CSV export in README", "CSV export: `shop.reports.to_csv(rows)`.\n"),
        bump("shop", "0.2.0", "0.3.0"),
    ]
    return Repo(
        name="culprit-test-n-plus-one",
        description="Fixture for Culprit: JOIN replaced by per-row queries (N+1)",
        expected_category="n_plus_one", acceptable_categories=["n_plus_one"],
        guilty=6, fixable=True, commits=commits,
    )


# --------------------------------------------------------------------------
# Repo B - lost cache (guilty: late)
# --------------------------------------------------------------------------

def repo_lost_cache() -> Repo:
    good_head = '"""Tax rules."""\nimport functools\n\n\n@functools.lru_cache(maxsize=None)\ndef load_tax_table(region):'
    bad_head = '"""Tax rules."""\n\n\ndef load_tax_table(region):'
    rules = d('''
        """Tax rules."""
        import functools


        @functools.lru_cache(maxsize=None)
        def load_tax_table(region):
            """Build the tax table for a region.

            Expensive: stands in for parsing a large rules file.
            """
            table = {}
            salt = sum(ord(ch) for ch in region)
            for i in range(20000):
                table[f"cat-{i}"] = ((i * 31 + salt) % 17) / 100.0
            return table


        def tax_rate(region, category_id):
            return load_tax_table(region).get(category_id, 0.0)
    ''')
    cart = d('''
        from pricing.rules import tax_rate


        def price_cart(items):
            """items: list of (region, category_id, net_amount)."""
            total = 0.0
            for region, category_id, net in items:
                total += net * (1 + tax_rate(region, category_id))
            return round(total, 2)
    ''')
    bench = d('''
        import time

        from pricing.cart import price_cart

        REGIONS = ["us-ca", "us-ny", "eu-de", "eu-fr", "uk", "jp", "in", "br"]


        def make_items(n=240):
            return [
                (REGIONS[i % len(REGIONS)], f"cat-{(i * 7) % 20000}", 10.0 + (i % 50))
                for i in range(n)
            ]


        def main():
            items = make_items()
            t0 = time.perf_counter()
            total = price_cart(items)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"cart total: {total}")
            print(f"{elapsed_ms:.2f}")


        if __name__ == "__main__":
            main()
    ''')
    first = common_files("pricing", "pricing-rules", "Tax lookup for cart pricing.")
    first.update({"pricing/rules.py": rules, "pricing/cart.py": cart, "bench.py": bench})
    commits = [
        CF("Initial commit: tax lookup and cart pricing", first),
        helper("Add region_label helper", "pricing/rules.py", '''
            def region_label(region):
                return region.upper().replace("-", " / ")
        '''),
        helper("Add round_money helper", "pricing/cart.py", '''
            def round_money(value):
                return round(value + 1e-9, 2)
        '''),
        readme_note("Document region codes", "Regions look like `us-ca`, `eu-de`, `uk`.\n"),
        bump("pricing", "0.1.0", "0.2.0"),
        helper("Add is_known_region helper", "pricing/rules.py", '''
            KNOWN_REGIONS = {"us-ca", "us-ny", "eu-de", "eu-fr", "uk", "jp", "in", "br"}


            def is_known_region(region):
                return region in KNOWN_REGIONS
        '''),
        helper("Add cart_size helper", "pricing/cart.py", '''
            def cart_size(items):
                return len(items)
        '''),
        readme_note("Add cart example to README", "```python\nprice_cart([(\"uk\", \"cat-1\", 10.0)])\n```\n"),
        bump("pricing", "0.2.0", "0.3.0"),
        CF("Drop lru_cache from load_tax_table to avoid serving stale tax data",
           {"pricing/rules.py": replace(good_head, bad_head)}),
        helper("Add net_total helper", "pricing/cart.py", '''
            def net_total(items):
                return round(sum(net for _, _, net in items), 2)
        '''),
        bump("pricing", "0.3.0", "0.3.1"),
    ]
    return Repo(
        name="culprit-test-lost-cache",
        description="Fixture for Culprit: memoisation removed from an expensive lookup",
        expected_category="lost_cache", acceptable_categories=["lost_cache"],
        guilty=9, fixable=True, commits=commits,
    )


# --------------------------------------------------------------------------
# Repo C - algorithmic complexity (guilty: FIRST commit after baseline)
# --------------------------------------------------------------------------

def repo_algorithmic() -> Repo:
    good = d('''
        def unique_user_ids(events):
            """Distinct user ids in first-seen order."""
            seen = set()
            result = []
            for event in events:
                uid = event["user_id"]
                if uid not in seen:
                    seen.add(uid)
                    result.append(uid)
            return result
    ''')
    bad = d('''
        def unique_user_ids(events):
            """Distinct user ids in first-seen order."""
            result = []
            for event in events:
                uid = event["user_id"]
                if uid not in result:
                    result.append(uid)
            return result
    ''')
    pipeline = d('''
        import json

        from events.dedupe import unique_user_ids


        def process(lines):
            events = [json.loads(line) for line in lines]
            return unique_user_ids(events)
    ''')
    bench = d('''
        import json
        import random
        import time

        from events.pipeline import process


        def make_lines(n=20000, users=4000, seed=3):
            rnd = random.Random(seed)
            return [
                json.dumps({"user_id": rnd.randint(1, users), "kind": "click", "ts": i})
                for i in range(n)
            ]


        def main():
            lines = make_lines()
            t0 = time.perf_counter()
            ids = process(lines)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"unique users: {len(ids)}")
            print(f"{elapsed_ms:.2f}")


        if __name__ == "__main__":
            main()
    ''')
    first = common_files("events", "event-dedupe", "Parse click events and list distinct users.")
    first.update({"events/dedupe.py": '"""Event de-duplication."""\n\n\n' + good, "events/pipeline.py": pipeline, "bench.py": bench})
    commits = [
        CF("Initial commit: event parsing and de-duplication", first),
        CF("Simplify unique_user_ids", {"events/dedupe.py": replace(good, bad)}),
        helper("Add count_events_by_user helper", "events/dedupe.py", '''
            def count_events_by_user(events):
                counts = {}
                for event in events:
                    counts[event["user_id"]] = counts.get(event["user_id"], 0) + 1
                return counts
        '''),
        helper("Add is_valid_event helper", "events/pipeline.py", '''
            def is_valid_event(event):
                return "user_id" in event and "ts" in event
        '''),
        readme_note("Document event shape", "Events are JSON objects with `user_id`, `kind`, `ts`.\n"),
        bump("events", "0.1.0", "0.2.0"),
        helper("Add first_event_ts helper", "events/pipeline.py", '''
            def first_event_ts(events):
                return min(e["ts"] for e in events) if events else None
        '''),
        bump("events", "0.2.0", "0.2.1"),
    ]
    return Repo(
        name="culprit-test-algorithmic-complexity",
        description="Fixture for Culprit: set-based dedupe replaced by O(n^2) list scan",
        expected_category="algorithmic_complexity", acceptable_categories=["algorithmic_complexity"],
        guilty=1, fixable=True, commits=commits,
    )


# --------------------------------------------------------------------------
# Repo D - blocking call (guilty: LAST commit == end_sha)
# --------------------------------------------------------------------------

def repo_blocking() -> Repo:
    good = d('''
        """Batch uploader."""
        import json

        BATCH_SIZE = 1000


        def upload_records(client, records):
            for start in range(0, len(records), BATCH_SIZE):
                batch = records[start:start + BATCH_SIZE]
                client.post(json.dumps(batch))
    ''')
    bad = d('''
        """Batch uploader."""
        import json
        import time

        BATCH_SIZE = 1000
        SETTLE_SECONDS = 0.02


        def upload_records(client, records):
            for start in range(0, len(records), BATCH_SIZE):
                batch = records[start:start + BATCH_SIZE]
                client.post(json.dumps(batch))
                time.sleep(SETTLE_SECONDS)  # give the receiving side a moment to settle
    ''')
    client = d('''
        class FakeClient:
            """Stands in for an HTTP client; just counts bytes."""

            def __init__(self):
                self.bytes_sent = 0
                self.requests = 0

            def post(self, body):
                self.bytes_sent += len(body)
                self.requests += 1
    ''')
    bench = d('''
        import time

        from syncer.client import FakeClient
        from syncer.uploader import upload_records


        def make_records(n=30000):
            return [
                {"id": i, "name": f"user-{i}", "tags": ["a", "b", "c"], "score": i * 0.5}
                for i in range(n)
            ]


        def main():
            records = make_records()
            client = FakeClient()
            t0 = time.perf_counter()
            upload_records(client, records)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"requests: {client.requests}")
            print(f"{elapsed_ms:.2f}")


        if __name__ == "__main__":
            main()
    ''')
    first = common_files("syncer", "record-syncer", "Uploads records to a (fake) server in batches.")
    first.update({"syncer/uploader.py": good, "syncer/client.py": client, "bench.py": bench})
    commits = [
        CF("Initial commit: batch uploader", first),
        helper("Add request_count helper", "syncer/client.py", '''
            def describe(client):
                return f"{client.requests} requests, {client.bytes_sent} bytes"
        '''),
        bump("syncer", "0.1.0", "0.2.0"),
        readme_note("Document batching", "Records are uploaded in batches of `BATCH_SIZE`.\n"),
        helper("Add chunked helper", "syncer/uploader.py", '''
            def chunked(items, size):
                for start in range(0, len(items), size):
                    yield items[start:start + size]
        '''),
        bump("syncer", "0.2.0", "0.2.1"),
        readme_note("Add FakeClient note", "`FakeClient` is used for local testing only.\n"),
        helper("Add total_bytes helper", "syncer/client.py", '''
            def total_bytes(client):
                return client.bytes_sent
        '''),
        CF("Add small settle delay between batches to be gentle on the receiver",
           {"syncer/uploader.py": replace(good, bad)}),
    ]
    return Repo(
        name="culprit-test-blocking-call",
        description="Fixture for Culprit: synchronous sleep added inside the upload loop",
        expected_category="blocking_call", acceptable_categories=["blocking_call"],
        guilty=8, fixable=True, commits=commits,
    )


# --------------------------------------------------------------------------
# Repo E - allocation overhead (guilty: middle, longer history)
# --------------------------------------------------------------------------

def repo_allocation() -> Repo:
    store = d('''
        """Item store."""


        class Store:
            def __init__(self):
                self._items = {}

            def add(self, item_id, name, price, qty, tags=None):
                self._items[item_id] = {
                    "id": item_id, "name": name, "price": price, "qty": qty,
                    "tags": list(tags or []),
                }

            def get(self, item_id):
                return self._items[item_id]

            def ids(self):
                return list(self._items)
    ''')
    valuation = d('''
        def total_value(store, rounds=60):
            total = 0.0
            for _ in range(rounds):
                for item_id in store.ids():
                    item = store.get(item_id)
                    total += item["price"] * item["qty"]
            return round(total, 2)
    ''')
    bench = d('''
        import time

        from inventory.store import Store
        from inventory.valuation import total_value


        def make_store(n=2000):
            store = Store()
            for i in range(n):
                store.add(i, f"item-{i}", 1.0 + (i % 40), 1 + (i % 9), ["t1", "t2"])
            return store


        def main():
            store = make_store()
            t0 = time.perf_counter()
            value = total_value(store)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"total value: {value}")
            print(f"{elapsed_ms:.2f}")


        if __name__ == "__main__":
            main()
    ''')
    old_head = '"""Item store."""\n\n\nclass Store:'
    new_head = '"""Item store."""\nimport copy\n\n\nclass Store:'
    old_get = "    def get(self, item_id):\n        return self._items[item_id]\n"
    new_get = ("    def get(self, item_id):\n"
               "        # hand out copies so callers can't mutate the store by accident\n"
               "        return copy.deepcopy(self._items[item_id])\n")

    def guilty_store(text: str) -> str:
        return replace(old_get, new_get)(replace(old_head, new_head)(text))

    first = common_files("inventory", "inventory-store", "In-memory item store with a valuation report.")
    first.update({"inventory/store.py": store, "inventory/valuation.py": valuation, "bench.py": bench})
    extras = "inventory/extras.py"
    commits = [
        CF("Initial commit: item store and valuation", first),
        helper("Add low_stock helper", extras, '''
            def low_stock(store, threshold=5):
                return [i for i in store.ids() if store.get(i)["qty"] < threshold]
        '''),
        bump("inventory", "0.1.0", "0.2.0"),
        helper("Add price_band helper", extras, '''
            def price_band(price):
                return "low" if price < 10 else "mid" if price < 30 else "high"
        '''),
        readme_note("Document Store API", "`Store.add`, `Store.get`, `Store.ids`.\n"),
        helper("Add tag_index helper", extras, '''
            def tag_index(store):
                index = {}
                for i in store.ids():
                    for tag in store.get(i)["tags"]:
                        index.setdefault(tag, []).append(i)
                return index
        '''),
        bump("inventory", "0.2.0", "0.2.1"),
        CF("Return copies from Store.get so callers cannot mutate stored items",
           {"inventory/store.py": guilty_store}),
        helper("Add restock helper", extras, '''
            def restock(store, item_id, amount):
                return store.get(item_id)["qty"] + amount
        '''),
        readme_note("Note on defensive copies", "`Store.get` returns a copy.\n"),
        helper("Add names helper", extras, '''
            def names(store):
                return sorted(store.get(i)["name"] for i in store.ids())
        '''),
        bump("inventory", "0.2.1", "0.3.0"),
    ]
    return Repo(
        name="culprit-test-allocation-overhead",
        description="Fixture for Culprit: deepcopy added to a hot read path",
        expected_category="allocation_overhead", acceptable_categories=["allocation_overhead"],
        guilty=7, fixable=True, commits=commits,
    )


# --------------------------------------------------------------------------
# Repo F - clean: NO regression anywhere in the range
# --------------------------------------------------------------------------

def repo_clean() -> Repo:
    good = d('''
        def word_freq(text):
            """Word -> count."""
            freq = {}
            for word in text.split():
                freq[word] = freq.get(word, 0) + 1
            return freq
    ''')
    faster = d('''
        def word_freq(text):
            """Word -> count."""
            return dict(Counter(text.split()))
    ''')
    bench = d('''
        import random
        import time

        from textstats.freq import word_freq

        VOCAB = [f"w{i}" for i in range(5000)]


        def make_text(n=250000, seed=5):
            rnd = random.Random(seed)
            return " ".join(rnd.choice(VOCAB) for _ in range(n))


        def main():
            text = make_text()
            t0 = time.perf_counter()
            freq = word_freq(text)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"distinct words: {len(freq)}")
            print(f"{elapsed_ms:.2f}")


        if __name__ == "__main__":
            main()
    ''')
    first = common_files("textstats", "textstats", "Word frequency counting.")
    first.update({"textstats/freq.py": '"""Word frequencies."""\n\n\n' + good, "bench.py": bench})
    commits = [
        CF("Initial commit: word frequency", first),
        helper("Add top_words helper", "textstats/freq.py", '''
            def top_words(freq, n=10):
                return sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
        '''),
        bump("textstats", "0.1.0", "0.2.0"),
        CF("Use collections.Counter for counting", {"textstats/freq.py": replace(
            '"""Word frequencies."""\n\n\n' + good,
            '"""Word frequencies."""\nfrom collections import Counter\n\n\n' + faster)}),
        readme_note("Document word_freq", "`word_freq(text)` returns `{word: count}`.\n"),
        helper("Add total_words helper", "textstats/freq.py", '''
            def total_words(freq):
                return sum(freq.values())
        '''),
        bump("textstats", "0.2.0", "0.2.1"),
        helper("Add unique_words helper", "textstats/freq.py", '''
            def unique_words(freq):
                return len(freq)
        '''),
    ]
    return Repo(
        name="culprit-test-no-regression",
        description="Fixture for Culprit: harmless changes only - expects NO regression found",
        expected_category=None, acceptable_categories=[],
        guilty=None, fixable=False, commits=commits,
    )


# --------------------------------------------------------------------------
# Repo G - "other": slowdown from a data-size constant, no code pattern
# --------------------------------------------------------------------------

def repo_other() -> Repo:
    config = "DEFAULT_ROWS = 20000\nSEED = 11\n"
    report = d('''
        import random

        from analytics.config import DEFAULT_ROWS, SEED

        REGIONS = ["north", "south", "east", "west", "central"]


        def build_sample(rows=DEFAULT_ROWS):
            rnd = random.Random(SEED)
            return [(rnd.choice(REGIONS), rnd.random() * 100) for _ in range(rows)]


        def summarize(data):
            totals = {}
            for region, value in data:
                totals[region] = totals.get(region, 0.0) + value
            return {r: round(v, 2) for r, v in totals.items()}


        def run_report(rows=DEFAULT_ROWS):
            return summarize(build_sample(rows))
    ''')
    bench = d('''
        import time

        from analytics.report import run_report


        def main():
            t0 = time.perf_counter()
            result = run_report()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"regions: {len(result)}")
            print(f"{elapsed_ms:.2f}")


        if __name__ == "__main__":
            main()
    ''')
    first = common_files("analytics", "analytics-report", "Regional totals over a generated sample dataset.")
    first.update({"analytics/config.py": config, "analytics/report.py": report, "bench.py": bench})
    commits = [
        CF("Initial commit: regional report", first),
        helper("Add region_share helper", "analytics/report.py", '''
            def region_share(totals):
                grand = sum(totals.values()) or 1.0
                return {r: round(v / grand, 4) for r, v in totals.items()}
        '''),
        bump("analytics", "0.1.0", "0.2.0"),
        readme_note("Document config", "Tune `analytics/config.py` for dataset size.\n"),
        helper("Add top_region helper", "analytics/report.py", '''
            def top_region(totals):
                return max(totals, key=totals.get)
        '''),
        CF("Expand default sample dataset to cover more traffic",
           {"analytics/config.py": replace("DEFAULT_ROWS = 20000", "DEFAULT_ROWS = 120000")}),
        bump("analytics", "0.2.0", "0.2.1"),
        readme_note("Note larger default dataset", "Default sample is now 120k rows.\n"),
    ]
    return Repo(
        name="culprit-test-other-cause",
        description="Fixture for Culprit: slowdown caused only by a bigger dataset constant (expects category 'other')",
        expected_category="other", acceptable_categories=["other"],
        guilty=5, fixable=False, commits=commits,
    )


ALL_REPOS: list[Callable[[], Repo]] = [
    repo_n_plus_one, repo_lost_cache, repo_algorithmic, repo_blocking,
    repo_allocation, repo_clean, repo_other,
]


# --------------------------------------------------------------------------
# git plumbing
# --------------------------------------------------------------------------

def git(repo_dir: Path, *args: str, env: dict | None = None) -> str:
    proc = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
        cwd=repo_dir, capture_output=True, text=True, env={**os.environ, **(env or {})},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo_dir}: {proc.stderr}")
    return proc.stdout.strip()


def build_repo(repo: Repo, out_dir: Path, force: bool) -> tuple[Path, list[str]]:
    repo_dir = out_dir / repo.name
    if repo_dir.exists():
        if not force:
            raise SystemExit(f"{repo_dir} already exists - pass --force to overwrite")
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True)
    git(repo_dir, "init", "-q", "-b", "main")
    git(repo_dir, "config", "user.name", AUTHOR_NAME)
    git(repo_dir, "config", "user.email", AUTHOR_EMAIL)

    shas: list[str] = []
    for i, commit in enumerate(repo.commits):
        for rel, transform in commit.files.items():
            path = repo_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            old = path.read_text() if path.exists() else ""
            new = transform(old) if callable(transform) else transform
            path.write_text(new)
        when = (T0 + timedelta(days=i)).isoformat()
        git(repo_dir, "add", "-A")
        git(repo_dir, "commit", "-q", "-m", commit.message,
            env={"GIT_AUTHOR_NAME": AUTHOR_NAME, "GIT_AUTHOR_EMAIL": AUTHOR_EMAIL,
                 "GIT_COMMITTER_NAME": AUTHOR_NAME, "GIT_COMMITTER_EMAIL": AUTHOR_EMAIL,
                 "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when})
        shas.append(git(repo_dir, "rev-parse", "HEAD"))
    return repo_dir, shas


# --------------------------------------------------------------------------
# validation: run the benchmark at every commit
# --------------------------------------------------------------------------

def run_bench(repo_dir: Path, runs: int) -> float:
    scores = []
    for _ in range(runs):
        proc = subprocess.run([sys.executable, "bench.py"], cwd=repo_dir,
                              capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"bench failed in {repo_dir.name}:\n{proc.stderr}")
        scores.append(float(proc.stdout.strip().splitlines()[-1]))
    return statistics.median(scores)


def validate(repo: Repo, repo_dir: Path, shas: list[str], runs: int) -> tuple[bool, list[float]]:
    print(f"\n  {'#':>2}  {'sha':7}  {'median ms':>10}  {'vs base':>8}  message")
    medians: list[float] = []
    for i, sha in enumerate(shas):
        git(repo_dir, "checkout", "-q", sha)
        medians.append(run_bench(repo_dir, runs))
    git(repo_dir, "checkout", "-q", "main")

    base = medians[0]
    ok = True
    problems: list[str] = []
    for i, (sha, ms) in enumerate(zip(shas, medians)):
        pct = (ms - base) / base * 100
        mark = ""
        if repo.guilty is not None and i == repo.guilty:
            mark = "  <-- GUILTY"
        print(f"  {i:>2}  {sha[:7]}  {ms:>10.2f}  {pct:>+7.1f}%  {repo.commits[i].message[:60]}{mark}")
        if i == 0:
            continue
        regressed_expected = repo.guilty is not None and i >= repo.guilty
        if regressed_expected:
            if ms < base * repo.min_ratio:
                problems.append(f"commit {i} should be >= {repo.min_ratio}x baseline but is {ms / base:.2f}x")
        else:
            if pct > 10.0:
                problems.append(f"commit {i} should be within 10% of baseline but is {pct:+.1f}%")
    if problems:
        ok = False
        print("  PROBLEMS:")
        for p in problems:
            print(f"   - {p}")
    return ok, medians


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="repos", help="output directory (default ./repos)")
    ap.add_argument("--only", help="build only repos whose name contains this text")
    ap.add_argument("--fast", action="store_true", help="1 bench run per commit instead of 3")
    ap.add_argument("--no-validate", action="store_true", help="skip the per-commit benchmark check")
    ap.add_argument("--force", action="store_true", help="overwrite existing repo directories")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = 1 if args.fast else 3

    manifest = []
    all_ok = True
    for factory in ALL_REPOS:
        repo = factory()
        if args.only and args.only not in repo.name:
            continue
        print(f"\n=== {repo.name}: {repo.description}")
        repo_dir, shas = build_repo(repo, out_dir, args.force)
        ok = True
        if not args.no_validate:
            ok, _ = validate(repo, repo_dir, shas, runs)
        all_ok &= ok
        manifest.append({
            "name": repo.name,
            "description": repo.description,
            "local_path": str(repo_dir),
            "benchmark_command": BENCH_CMD,
            "start_sha": shas[0],
            "end_sha": shas[-1],
            "commit_count": len(shas) - 1,
            "guilty_sha": shas[repo.guilty] if repo.guilty is not None else None,
            "guilty_message": repo.commits[repo.guilty].message if repo.guilty is not None else None,
            "expected_regression": repo.guilty is not None,
            "expected_category": repo.expected_category,
            "acceptable_categories": repo.acceptable_categories,
            "fix_expected": repo.fixable,
            "validated": ok if not args.no_validate else None,
        })

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {out_dir / 'manifest.json'} ({len(manifest)} repos)")
    if not args.no_validate:
        print("ALL REPOS VALIDATED" if all_ok else "SOME REPOS FAILED VALIDATION - see PROBLEMS above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
