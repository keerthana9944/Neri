"""
NERI Industrial Maintenance Assistant - Automated Load Testing Suite
Locust load test scenarios simulating realistic industrial field usage:
- Health pings (monitoring/infrastructure)
- Technical documentation retrieval
- Troubleshooting history lookups
- Safe cached/active troubleshooting queries
- Fast fail-closed authentication rejection verification
"""

import os
import random
from pathlib import Path
from dotenv import load_dotenv
from locust import HttpUser, task, between, events

# Load local environment variables if available
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("NERI_TECHNICIAN_KEY", "neri-technician-secret-key-2026")
SUPERVISOR_KEY = os.getenv("NERI_SUPERVISOR_KEY", "neri-supervisor-secret-key-2026")


class TechnicianUser(HttpUser):
    """
    Simulates a factory maintenance technician querying manuals, history,
    and diagnostic troubleshooting advice.
    """
    wait_time = between(0.5, 2.0)

    def on_start(self):
        """Configure technician authentication headers."""
        self.headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json",
        }

    @task(4)
    def check_health(self):
        """Simulate lightweight health checks / liveness probes."""
        with self.client.get(
            "/health",
            name="GET /health",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(3)
    def get_documents_catalog(self):
        """Simulate browsing approved machinery documentation."""
        with self.client.get(
            "/documents",
            headers=self.headers,
            name="GET /documents",
            catch_response=True,
        ) as response:
            if response.status_code == 200 and "documents" in response.text:
                response.success()
            else:
                response.failure(f"Documents fetch failed: {response.status_code}")

    @task(2)
    def get_troubleshooting_history(self):
        """Simulate retrieving historical repair cases."""
        with self.client.get(
            "/history",
            headers=self.headers,
            name="GET /history",
            catch_response=True,
        ) as response:
            if response.status_code == 200 and "sessions" in response.text:
                response.success()
            else:
                response.failure(f"History fetch failed: {response.status_code}")

    @task(1)
    def post_troubleshooting_query(self):
        """Simulate submitting a diagnostic request for an industrial machine."""
        sample_queries = [
            {
                "machine": "CNC Mill",
                "machine_id": "CNC-001",
                "problem": "Spindle motor overheating warning code E-401",
                "error_code": "E-401",
            },
            {
                "machine": "Hydraulic Press",
                "machine_id": "HYD-002",
                "problem": "Hydraulic pressure loss during cycle execution",
                "error_code": "P-102",
            },
        ]
        payload = random.choice(sample_queries)
        with self.client.post(
            "/query",
            json=payload,
            headers=self.headers,
            name="POST /query",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Query failed: {response.status_code} - {response.text[:100]}")

    @task(1)
    def test_unauthorized_rejection(self):
        """Verify API quickly rejects unauthorized requests (fail-closed integrity)."""
        bad_headers = {"X-API-Key": "invalid-token-test"}
        with self.client.get(
            "/documents",
            headers=bad_headers,
            name="GET /documents (Unauthorized)",
            catch_response=True,
        ) as response:
            if response.status_code == 401:
                response.success()
            else:
                response.failure(f"Expected 401 for bad token, got: {response.status_code}")


class MonitoringProbeUser(HttpUser):
    """
    Simulates high-frequency infrastructure health monitoring and uptime probes.
    """
    wait_time = between(0.2, 1.0)

    @task
    def probe_root(self):
        self.client.get("/", name="GET / (Root)")

    @task
    def probe_health(self):
        self.client.get("/health", name="GET /health (Monitoring)")
