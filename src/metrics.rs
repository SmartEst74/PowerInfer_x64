//! Prometheus metrics for PowerInfer
//!
//! Exposes metrics on `/metrics` endpoint in Prometheus exposition format.
//! All metrics here must match the alerting rules in `infrastructure/alerting/rules.yml`.

use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry,
    TextEncoder,
};
use std::sync::Arc;

/// Application metrics registry
#[derive(Clone)]
pub struct Metrics {
    registry: Registry,

    /// Total inference requests by status code
    pub inference_requests_total: IntCounterVec,

    /// Inference latency histogram (seconds)
    pub inference_duration_seconds: HistogramVec,

    /// Tokens generated total
    pub tokens_generated_total: IntCounterVec,

    /// Current queue depth
    pub queue_depth: IntGauge,

    /// GPU memory usage bytes by device
    pub gpu_memory_usage_bytes: IntGaugeVec,

    /// GPU memory total bytes by device
    pub gpu_memory_total_bytes: IntGaugeVec,

    /// GPU utilization percent by device
    pub gpu_utilization_percent: IntGaugeVec,

    /// Model info gauge (constant 1 with labels)
    pub model_info: IntGaugeVec,

    /// Server uptime seconds
    pub uptime_seconds: IntGauge,
}

impl Metrics {
    /// Create a new metrics registry with all PowerInfer metrics
    pub fn new() -> Self {
        let registry = Registry::new();

        let inference_requests_total = IntCounterVec::new(
            Opts::new(
                "powerinfer_inference_requests_total",
                "Total number of inference requests",
            ),
            &["endpoint", "status"],
        )
        .expect("failed to create inference_requests_total metric");

        let inference_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "powerinfer_inference_duration_seconds",
                "Inference request latency in seconds",
            )
            .buckets(vec![
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ]),
            &["endpoint"],
        )
        .expect("failed to create inference_duration_seconds metric");

        let tokens_generated_total = IntCounterVec::new(
            Opts::new(
                "powerinfer_tokens_generated_total",
                "Total tokens generated across all requests",
            ),
            &["model"],
        )
        .expect("failed to create tokens_generated_total metric");

        let queue_depth = IntGauge::new("powerinfer_queue_depth", "Current request queue depth")
            .expect("failed to create queue_depth metric");

        let gpu_memory_usage_bytes = IntGaugeVec::new(
            Opts::new(
                "powerinfer_gpu_memory_usage_bytes",
                "GPU memory usage in bytes",
            ),
            &["device"],
        )
        .expect("failed to create gpu_memory_usage_bytes metric");

        let gpu_memory_total_bytes = IntGaugeVec::new(
            Opts::new(
                "powerinfer_gpu_memory_total_bytes",
                "Total GPU memory in bytes",
            ),
            &["device"],
        )
        .expect("failed to create gpu_memory_total_bytes metric");

        let gpu_utilization_percent = IntGaugeVec::new(
            Opts::new(
                "powerinfer_gpu_utilization_percent",
                "GPU compute utilization percentage",
            ),
            &["device"],
        )
        .expect("failed to create gpu_utilization_percent metric");

        let model_info = IntGaugeVec::new(
            Opts::new("powerinfer_model_info", "Model information (always 1)"),
            &["name", "arch", "quantization"],
        )
        .expect("failed to create model_info metric");

        let uptime_seconds = IntGauge::new("powerinfer_uptime_seconds", "Server uptime in seconds")
            .expect("failed to create uptime_seconds metric");

        registry
            .register(Box::new(inference_requests_total.clone()))
            .expect("failed to register inference_requests_total");
        registry
            .register(Box::new(inference_duration_seconds.clone()))
            .expect("failed to register inference_duration_seconds");
        registry
            .register(Box::new(tokens_generated_total.clone()))
            .expect("failed to register tokens_generated_total");
        registry
            .register(Box::new(queue_depth.clone()))
            .expect("failed to register queue_depth");
        registry
            .register(Box::new(gpu_memory_usage_bytes.clone()))
            .expect("failed to register gpu_memory_usage_bytes");
        registry
            .register(Box::new(gpu_memory_total_bytes.clone()))
            .expect("failed to register gpu_memory_total_bytes");
        registry
            .register(Box::new(gpu_utilization_percent.clone()))
            .expect("failed to register gpu_utilization_percent");
        registry
            .register(Box::new(model_info.clone()))
            .expect("failed to register model_info");
        registry
            .register(Box::new(uptime_seconds.clone()))
            .expect("failed to register uptime_seconds");

        Self {
            registry,
            inference_requests_total,
            inference_duration_seconds,
            tokens_generated_total,
            queue_depth,
            gpu_memory_usage_bytes,
            gpu_memory_total_bytes,
            gpu_utilization_percent,
            model_info,
            uptime_seconds,
        }
    }

    /// Render all metrics in Prometheus exposition format
    pub fn gather(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder
            .encode(&metric_families, &mut buffer)
            .expect("failed to encode metrics");
        String::from_utf8(buffer).expect("metrics output was not valid UTF-8")
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared metrics handle
pub type SharedMetrics = Arc<Metrics>;
