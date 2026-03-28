/**
 * PowerInfer_x64 SLO Validation Load Test
 *
 * Run with: k6 run tests/load/slo-validation.js
 *
 * Validates SLOs defined in README:
 *   - Availability: 99.9%
 *   - Latency p50: <50ms
 *   - Latency p99: <500ms
 *   - Throughput: >10 req/s
 *   - Error Rate: <0.1%
 */

import http from "k6/http";
import { check, sleep } from "k6";
import { Rate, Trend } from "k6/metrics";

// Custom metrics
const errorRate = new Rate("errors");
const latencyP50 = new Trend("latency_p50", true);
const latencyP99 = new Trend("latency_p99", true);

// SLO thresholds
export const options = {
  thresholds: {
    // SLO: Error rate < 0.1%
    errors: ["rate<0.001"],
    // SLO: p50 latency < 50ms
    latency_p50: ["p(50)<50"],
    // SLO: p99 latency < 500ms
    latency_p99: ["p(99)<500"],
    // SLO: HTTP req duration < 500ms for 99% of requests
    http_req_duration: ["p(99)<500"],
    // SLO: Throughput > 10 req/s (checked via iteration rate)
    http_reqs: ["rate>10"],
  },
  stages: [
    // Warm up
    { duration: "30s", target: 5 },
    // Ramp to steady state
    { duration: "1m", target: 20 },
    // Sustained load
    { duration: "2m", target: 20 },
    // Peak load
    { duration: "30s", target: 50 },
    // Cool down
    { duration: "30s", target: 0 },
  ],
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080";

export default function () {
  // Health check (10% of traffic)
  if (Math.random() < 0.1) {
    const healthRes = http.get(`${BASE_URL}/health`);
    check(healthRes, {
      "health status 200": (r) => r.status === 200,
      "health response OK": (r) => r.body === "OK",
    });
    errorRate.add(healthRes.status !== 200);
    return;
  }

  // Completions endpoint (60% of traffic)
  if (Math.random() < 0.6) {
    const payload = JSON.stringify({
      model: "powerinfer-model",
      prompt: "Hello, how are you?",
      max_tokens: 50,
      temperature: 0.7,
    });

    const res = http.post(`${BASE_URL}/v1/completions`, payload, {
      headers: { "Content-Type": "application/json" },
    });

    const passed = check(res, {
      "completions status 200": (r) => r.status === 200,
      "completions has choices": (r) => {
        try {
          return JSON.parse(r.body).choices.length > 0;
        } catch {
          return false;
        }
      },
      "completions latency OK": (r) => r.timings.duration < 500,
    });

    errorRate.add(!passed);
    latencyP50.add(res.timings.duration);
    latencyP99.add(res.timings.duration);
  }
  // Chat completions (30% of traffic)
  else {
    const payload = JSON.stringify({
      model: "powerinfer-model",
      messages: [
        { role: "user", content: "What is the capital of France?" },
      ],
      max_tokens: 50,
      temperature: 0.7,
    });

    const res = http.post(`${BASE_URL}/v1/chat/completions`, payload, {
      headers: { "Content-Type": "application/json" },
    });

    const passed = check(res, {
      "chat status 200": (r) => r.status === 200,
      "chat has choices": (r) => {
        try {
          return JSON.parse(r.body).choices.length > 0;
        } catch {
          return false;
        }
      },
      "chat latency OK": (r) => r.timings.duration < 500,
    });

    errorRate.add(!passed);
    latencyP50.add(res.timings.duration);
    latencyP99.add(res.timings.duration);
  }

  sleep(0.1);
}

export function handleSummary(data) {
  const passed = Object.values(data.root_group.checks).every(
    (c) => c.fails === 0
  );

  console.log("\n=== SLO Validation Summary ===");
  console.log(`Status: ${passed ? "PASS" : "FAIL"}`);
  console.log(
    `Error rate: ${(data.metrics.errors?.values?.rate * 100 || 0).toFixed(3)}%`
  );
  console.log(
    `p50 latency: ${data.metrics.http_req_duration?.values?.["p(50)"] || "N/A"}ms`
  );
  console.log(
    `p99 latency: ${data.metrics.http_req_duration?.values?.["p(99)"] || "N/A"}ms`
  );
  console.log(
    `Throughput: ${data.metrics.http_reqs?.values?.rate?.toFixed(1) || "N/A"} req/s`
  );

  return {
    stdout: JSON.stringify(data, null, 2),
  };
}
