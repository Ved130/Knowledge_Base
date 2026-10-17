import argparse
import json
import statistics
import time
 
import httpx
 
 
def parse_server_timing(header: str) -> dict:
    out = {}
    for part in header.split(","):
        part = part.strip()
        if ";dur=" in part:
            name, dur = part.split(";dur=")
            out[name.strip()] = float(dur)
    return out
 
 
def percentile(values, p):
    if len(values) < 2:
        return values[0] if values else None
    return statistics.quantiles(values, n=100, method="inclusive")[p - 1]
 
 
def clear_cache(redis_url, prefix):
    import redis
 
    r = redis.from_url(redis_url)
    deleted = 0
    for key in r.scan_iter(match=f"{prefix}*"):
        r.delete(key)
        deleted += 1
    print(f"Cleared {deleted} cached keys with prefix '{prefix}'")
 
 
def run_pass(client, url, headers, field, questions, label, sleep):
    rows = []
    failures = 0
    print(f"\n--- {label} pass: {len(questions)} questions ---")
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        try:
            resp = client.post(url, json={field: q}, headers=headers)
        except httpx.HTTPError as e:
            failures += 1
            print(f"[{label}] {i}/{len(questions)} ERROR {type(e).__name__}: {q[:50]}")
            continue
        client_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200:
            failures += 1
            print(f"[{label}] {i}/{len(questions)} FAILED {resp.status_code}: {q[:50]} | {resp.text[:120]}")
            time.sleep(sleep)
            continue
        rows.append(
            {
                "question": q,
                "client_ms": round(client_ms, 2),
                "cache": resp.headers.get("X-Cache", "?"),
                "stages": parse_server_timing(resp.headers.get("Server-Timing", "")),
            }
        )
        r = rows[-1]
        llm = r["stages"].get("llm")
        llm_txt = f" llm={llm:.0f}ms" if llm is not None else ""
        print(f"[{label}] {i}/{len(questions)} {r['cache']:<4} {r['client_ms']:>8.0f} ms{llm_txt}  {q[:45]}")
        time.sleep(sleep)
    if failures:
        print(f"[{label}] {failures} request(s) failed and were excluded")
    return rows
 
 
def summarise(rows, label):
    if not rows:
        print(f"\n{label}: no successful requests")
        return {}
    summary = {"n": len(rows)}
    client = [r["client_ms"] for r in rows]
    summary["client_ms"] = {"p50": percentile(client, 50), "p95": percentile(client, 95)}
 
    stage_names = sorted({k for r in rows for k in r["stages"]})
    for s in stage_names:
        vals = [r["stages"][s] for r in rows if s in r["stages"]]
        summary[s] = {"p50": percentile(vals, 50), "p95": percentile(vals, 95)}
 
    hits = sum(1 for r in rows if r["cache"] == "HIT")
    summary["cache_hit_rate"] = round(hits / len(rows) * 100, 1)
 
    print(f"\n=== {label} (n={len(rows)}, cache hits {summary['cache_hit_rate']}%) ===")
    print(f"{'metric':<16}{'p50 ms':>10}{'p95 ms':>10}")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"{k:<16}{v['p50']:>10.1f}{v['p95']:>10.1f}")
    return summary
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--key-header", default="X-API-Key")
    ap.add_argument("--field", default="question", help="JSON field name for the query")
    ap.add_argument("--questions", default="questions.txt")
    ap.add_argument("--redis-url", default=None)
    ap.add_argument("--cache-prefix", default="")
    ap.add_argument("--sleep", type=float, default=3.0, help="seconds between requests (not timed)")
    args = ap.parse_args()
 
    with open(args.questions, encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
 
    headers = {args.key_header: args.api_key} if args.api_key else {}
 
    if args.redis_url:
        clear_cache(args.redis_url, args.cache_prefix)
    else:
        print("WARNING: no --redis-url given; pass 1 may include cache hits.")
 
    with httpx.Client(timeout=120) as client:
        cold = run_pass(client, args.url, headers, args.field, questions, "cold", args.sleep)
        warm = run_pass(client, args.url, headers, args.field, questions, "warm", args.sleep)
 
    results = {
        "cold": summarise(cold, "Cold (cache miss)"),
        "warm": summarise(warm, "Warm (cache hit)"),
        "raw": {"cold": cold, "warm": warm},
    }
 
    c, w = results["cold"].get("client_ms"), results["warm"].get("client_ms")
    if c and w:
        drop = (1 - w["p50"] / c["p50"]) * 100
        print(f"\nMedian latency: {c['p50']:.0f} ms -> {w['p50']:.0f} ms ({drop:.0f}% reduction)")
 
    with open("latency_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved latency_results.json")
 
 
if __name__ == "__main__":
    main()