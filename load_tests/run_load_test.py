"""
NERI Industrial Maintenance Assistant - Standalone Automated Load Test Runner

A lightweight, dependency-minimal benchmark harness for validating API latency,
throughput, and fail-closed security under concurrent technician load.

Features:
- Multi-threaded concurrent worker execution
- Realistic task distributions (health, documents, history, query, security)
- Detailed percentile calculations (p50, p90, p95, p99, min, max, average)
- Formatted markdown report output
- SLA assertions and exit codes for CI/CD integration
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class LoadTestRunner:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        users: int = 5,
        duration: int = 10,
        warmup_seconds: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.users = users
        self.duration = duration
        self.warmup_seconds = warmup_seconds
        self.results: List[Dict] = []

    def _execute_single_request(self, session: requests.Session) -> Optional[Dict]:
        """Perform a single weighted endpoint request and record latency."""
        roll = random.random()
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        # Weighted task distribution
        if roll < 0.35:
            # 35% Health ping
            endpoint = "/health"
            method = "GET"
            name = "GET /health"
            target_url = f"{self.base_url}/health"
            expected_status = 200
            req_headers = {}
            payload = None
        elif roll < 0.60:
            # 25% Documents catalog
            endpoint = "/documents"
            method = "GET"
            name = "GET /documents"
            target_url = f"{self.base_url}/documents"
            expected_status = 200
            req_headers = headers
            payload = None
        elif roll < 0.80:
            # 20% History retrieval
            endpoint = "/history"
            method = "GET"
            name = "GET /history"
            target_url = f"{self.base_url}/history"
            expected_status = 200
            req_headers = headers
            payload = None
        elif roll < 0.90:
            # 10% Troubleshooting query
            endpoint = "/query"
            method = "POST"
            name = "POST /query"
            target_url = f"{self.base_url}/query"
            expected_status = 200
            req_headers = headers
            payload = {
                "machine": "CNC Mill",
                "machine_id": "CNC-001",
                "problem": "Spindle motor overheating warning code E-401",
                "error_code": "E-401",
            }
        else:
            # 10% Auth rejection check (fail-closed integrity)
            endpoint = "/documents"
            method = "GET"
            name = "GET /documents (Unauthorized Check)"
            target_url = f"{self.base_url}/documents"
            expected_status = 401
            req_headers = {"X-API-Key": "invalid-unauthorized-probe"}
            payload = None

        start_time = time.perf_counter()
        status_code = 0
        success = False
        error_msg = ""

        try:
            if method == "GET":
                resp = session.get(target_url, headers=req_headers, timeout=10.0)
            else:
                resp = session.post(target_url, json=payload, headers=req_headers, timeout=15.0)

            status_code = resp.status_code
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            success = (status_code == expected_status)
            if not success:
                error_msg = f"Status {status_code} != expected {expected_status}"
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            error_msg = str(exc)

        return {
            "name": name,
            "endpoint": endpoint,
            "method": method,
            "latency_ms": latency_ms,
            "status_code": status_code,
            "success": success,
            "error": error_msg,
        }

    def _worker_loop(self, deadline: float) -> List[Dict]:
        """Worker thread executing requests until the deadline expires."""
        worker_results = []
        with requests.Session() as session:
            while time.perf_counter() < deadline:
                rec = self._execute_single_request(session)
                if rec:
                    worker_results.append(rec)
                # Small jitter between 10ms and 50ms to simulate network think-time
                time.sleep(random.uniform(0.01, 0.05))
        return worker_results

    def run(self) -> Dict:
        """Run the multi-threaded load test and compute benchmarks."""
        print(f"[*] Starting NERI Load Test Benchmark")
        print(f"    Target Host:   {self.base_url}")
        print(f"    Concurrency:   {self.users} users")
        print(f"    Test Duration: {self.duration} seconds")
        print(f"    API Key:       {self.api_key[:6]}***{self.api_key[-4:] if len(self.api_key) > 10 else ''}")

        # Quick connectivity pre-flight check
        try:
            preflight = requests.get(f"{self.base_url}/health", timeout=3.0)
            if preflight.status_code != 200:
                print(f"[!] Warning: Preflight /health returned status {preflight.status_code}")
        except Exception as exc:
            print(f"[x] Preflight connectivity error: {exc}")
            raise

        deadline = time.perf_counter() + self.duration
        start_overall = time.perf_counter()

        all_records: List[Dict] = []
        with ThreadPoolExecutor(max_workers=self.users) as executor:
            futures = [executor.submit(self._worker_loop, deadline) for _ in range(self.users)]
            for fut in as_completed(futures):
                all_records.extend(fut.result())

        total_elapsed = time.perf_counter() - start_overall
        return self._calculate_metrics(all_records, total_elapsed)

    def _calculate_metrics(self, records: List[Dict], total_time: float) -> Dict:
        """Calculate statistics, percentiles, and RPS."""
        if not records:
            return {"total_requests": 0, "error": "No requests recorded"}

        def percentile(values: List[float], p: float) -> float:
            if not values:
                return 0.0
            idx = int(round((len(values) - 1) * p))
            return values[idx]

        total_count = len(records)
        success_count = sum(1 for r in records if r["success"])
        fail_count = total_count - success_count
        rps = total_count / total_time if total_time > 0 else 0.0

        # Group by endpoint
        by_endpoint: Dict[str, List[Dict]] = {}
        for r in records:
            by_endpoint.setdefault(r["name"], []).append(r)

        endpoint_stats = {}
        for ep_name, ep_records in sorted(by_endpoint.items()):
            latencies = sorted(r["latency_ms"] for r in ep_records)
            ep_success = sum(1 for r in ep_records if r["success"])
            endpoint_stats[ep_name] = {
                "count": len(ep_records),
                "success": ep_success,
                "failed": len(ep_records) - ep_success,
                "rps": round(len(ep_records) / total_time, 2),
                "avg_ms": round(sum(latencies) / len(latencies), 2),
                "p50_ms": round(percentile(latencies, 0.50), 2),
                "p90_ms": round(percentile(latencies, 0.90), 2),
                "p95_ms": round(percentile(latencies, 0.95), 2),
                "p99_ms": round(percentile(latencies, 0.99), 2),
                "min_ms": round(latencies[0], 2),
                "max_ms": round(latencies[-1], 2),
            }

        all_latencies = sorted(r["latency_ms"] for r in records)
        summary = {
            "total_requests": total_count,
            "successful_requests": success_count,
            "failed_requests": fail_count,
            "failure_rate_percent": round((fail_count / total_count) * 100.0, 2),
            "duration_seconds": round(total_time, 2),
            "requests_per_second": round(rps, 2),
            "latency": {
                "min_ms": round(all_latencies[0], 2),
                "avg_ms": round(sum(all_latencies) / len(all_latencies), 2),
                "p50_ms": round(percentile(all_latencies, 0.50), 2),
                "p90_ms": round(percentile(all_latencies, 0.90), 2),
                "p95_ms": round(percentile(all_latencies, 0.95), 2),
                "p99_ms": round(percentile(all_latencies, 0.99), 2),
                "max_ms": round(all_latencies[-1], 2),
            },
            "by_endpoint": endpoint_stats,
        }
        return summary


def print_report(metrics: Dict):
    """Print readable report with ASCII/Markdown table."""
    print("\n" + "=" * 80)
    print("                    NERI LOAD TEST BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total Requests:     {metrics['total_requests']}")
    print(f"Success / Failed:   {metrics['successful_requests']} / {metrics['failed_requests']} ({metrics['failure_rate_percent']}% error rate)")
    print(f"Benchmark Duration: {metrics['duration_seconds']}s")
    print(f"Throughput (RPS):   {metrics['requests_per_second']} req/sec")
    print(f"Latency Percentiles: p50={metrics['latency']['p50_ms']}ms | p90={metrics['latency']['p90_ms']}ms | p95={metrics['latency']['p95_ms']}ms | p99={metrics['latency']['p99_ms']}ms")
    print("-" * 80)
    print(f"{'Endpoint':<35} {'Reqs':<6} {'RPS':<7} {'Avg':<8} {'p50':<8} {'p95':<8} {'p99':<8}")
    print("-" * 80)
    for name, s in metrics["by_endpoint"].items():
        print(f"{name:<35} {s['count']:<6} {s['rps']:<7} {s['avg_ms']:<8.1f} {s['p50_ms']:<8.1f} {s['p95_ms']:<8.1f} {s['p99_ms']:<8.1f}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="NERI Standalone API Load Test Runner")
    parser.add_argument("--host", default=os.getenv("RAG_API_URL", "http://127.0.0.1:8000"), help="API base URL")
    parser.add_argument("--users", type=int, default=5, help="Number of concurrent user threads")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds")
    parser.add_argument("--api-key", default=os.getenv("NERI_TECHNICIAN_KEY", "neri-technician-secret-key-2026"), help="Technician API Key")
    parser.add_argument("--max-fail-rate", type=float, default=5.0, help="Maximum allowed failure rate percentage")
    parser.add_argument("--output", default=None, help="Save metrics to JSON file path")
    parser.add_argument("--dry-run", action="store_true", help="Perform mock benchmark calculation without live network")

    args = parser.parse_args()

    if args.dry_run:
        print("[*] Running dry-run validation...")
        mock_records = [
            {"name": "GET /health", "endpoint": "/health", "method": "GET", "latency_ms": 4.2, "status_code": 200, "success": True, "error": ""},
            {"name": "GET /documents", "endpoint": "/documents", "method": "GET", "latency_ms": 12.5, "status_code": 200, "success": True, "error": ""},
            {"name": "GET /history", "endpoint": "/history", "method": "GET", "latency_ms": 14.1, "status_code": 200, "success": True, "error": ""},
        ]
        runner = LoadTestRunner(base_url="http://mock", api_key="mock", users=1, duration=1)
        res = runner._calculate_metrics(mock_records, total_time=1.0)
        print_report(res)
        sys.exit(0)

    try:
        runner = LoadTestRunner(
            base_url=args.host,
            api_key=args.api_key,
            users=args.users,
            duration=args.duration,
        )
        metrics = runner.run()
        print_report(metrics)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(metrics, indent=2))
            print(f"[+] Benchmark results exported to {out_path}")

        # Assert SLA threshold
        if metrics["failure_rate_percent"] > args.max_fail_rate:
            print(f"[!] ERROR: Failure rate {metrics['failure_rate_percent']}% exceeds SLA threshold {args.max_fail_rate}%")
            sys.exit(1)

        print("[+] Load test completed successfully within SLA thresholds.")
        sys.exit(0)

    except Exception as exc:
        print(f"[!] Load test execution error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
