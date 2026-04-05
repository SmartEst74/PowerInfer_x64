//! OpenAI-compatible HTTP API server
//!
//! Provides /v1/completions, /v1/chat/completions, /v1/models, /health, and /metrics endpoints
//! using Axum and Tokio.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

use crate::metrics::{Metrics, SharedMetrics};
use crate::{GenerationOptions, InferenceContext};

/// Application state shared across requests
#[derive(Clone)]
pub struct AppState {
    pub model: Arc<Mutex<InferenceContext>>,
    pub max_concurrent: usize,
    pub semaphore: Arc<Semaphore>,
    pub metrics: SharedMetrics,
    pub start_epoch: u64,
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_path: String,
    pub max_concurrent: usize,
    pub max_queue_depth: usize,
    pub hot_index_path: Option<String>,
}

/// OpenAI-style request: Completions
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
}

/// OpenAI-style response: Completions
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

/// OpenAI-style response: Chat completions
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatCompletionMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: String,
}

/// OpenAI-style request: Chat completions
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

fn request_token_budget(max_tokens: Option<usize>) -> usize {
    max_tokens.unwrap_or(64).clamp(1, 512)
}

fn build_chat_prompt(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|message| format!("{}: {}", message.role, message.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn request_generation_options(
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
) -> GenerationOptions {
    let seed = chrono::Utc::now()
        .timestamp_nanos_opt()
        .map(|value| value as u64)
        .unwrap_or_else(|| chrono::Utc::now().timestamp_micros() as u64);

    GenerationOptions {
        max_tokens: request_token_budget(max_tokens),
        temperature: temperature.unwrap_or(0.0),
        top_p: top_p.unwrap_or(1.0),
        seed,
    }
}

fn validate_generation_options(
    stream: Option<bool>,
    temperature: Option<f32>,
    top_p: Option<f32>,
) -> Result<(), ApiError> {
    if stream.unwrap_or(false) {
        return Err(ApiError::bad_request(
            "streaming responses are not implemented",
        ));
    }

    if let Some(temperature) = temperature {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(ApiError::bad_request(
                "temperature must be a finite value greater than or equal to 0.0",
            ));
        }
    }

    if let Some(top_p) = top_p {
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(ApiError::bad_request(
                "top_p must be a finite value in the range (0.0, 1.0]",
            ));
        }
    }

    Ok(())
}

fn make_completion_response(model: String, text: String) -> CompletionResponse {
    CompletionResponse {
        id: format!("cmpl-{}", chrono::Utc::now().timestamp_millis()),
        object: "text_completion".to_string(),
        created: chrono::Utc::now().timestamp() as u64,
        model,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
    }
}

fn make_chat_response(model: String, text: String) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: format!("chatcmpl-{}", chrono::Utc::now().timestamp_millis()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp() as u64,
        model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_string(),
                content: text,
            },
            finish_reason: "stop".to_string(),
        }],
    }
}

/// API error type for axum handlers
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({
                "error": {
                    "message": self.message,
                    "type": if self.status == StatusCode::BAD_REQUEST {
                        "invalid_request_error"
                    } else {
                        "server_error"
                    }
                }
            })),
        )
            .into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: err.to_string(),
        }
    }
}

impl From<tokio::sync::AcquireError> for ApiError {
    fn from(err: tokio::sync::AcquireError) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: err.to_string(),
        }
    }
}

/// Build Axum router with all endpoints
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/completions", post(handle_completions))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/models", get(handle_list_models))
        .route("/health", get(handle_health))
        .route("/metrics", get(handle_metrics))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state)
}

/// Start the HTTP server
pub async fn serve(config: ServerConfig) -> anyhow::Result<()> {
    info!("Loading model from: {}", config.model_path);
    let mut ctx =
        InferenceContext::from_gguf(&config.model_path, crate::runtime::BackendFactory::cpu())?;

    if let Some(hot_index_path) = &config.hot_index_path {
        let index = crate::activation::HotNeuronIndex::load(hot_index_path)?;
        let indexed_layers = index.layers.len();
        let indexed_units: usize = index
            .layers
            .iter()
            .map(|layer| layer.hot_indices.len())
            .sum();
        ctx.set_hot_index(index)?;
        info!(
            "Loaded hot index from {} ({} layers, {} tracked units)",
            hot_index_path, indexed_layers, indexed_units
        );
    }

    info!(
        "Model loaded: {} ({} layers)",
        ctx.config().name.as_deref().unwrap_or("unknown"),
        ctx.config().block_count
    );

    let metrics = Arc::new(Metrics::new());
    let model_name = ctx
        .config()
        .name
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    metrics
        .model_info
        .with_label_values(&[
            &model_name,
            &ctx.config().arch,
            &format!("{:?}", ctx.config().quantization),
        ])
        .set(1);

    let ctx = Arc::new(Mutex::new(ctx));
    let max_concurrent = if config.max_concurrent == 0 {
        1
    } else {
        config.max_concurrent
    };
    let semaphore = Arc::new(Semaphore::new(max_concurrent));

    let state = AppState {
        model: ctx,
        max_concurrent,
        semaphore,
        metrics,
        start_epoch: chrono::Utc::now().timestamp() as u64,
    };

    let router = build_router(state);
    let addr: std::net::SocketAddr = format!("{}:{}", config.host, config.port).parse()?;

    info!("PowerInfer_x64 server listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}

/// Handle /v1/completions
async fn handle_completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, ApiError> {
    validate_generation_options(req.stream, req.temperature, req.top_p)?;

    let timer = state
        .metrics
        .inference_duration_seconds
        .with_label_values(&["completions"])
        .start_timer();

    state.metrics.queue_depth.inc();
    let _permit = state.semaphore.clone().acquire_owned().await?;
    state.metrics.queue_depth.dec();

    let text = {
        let mut model = state.model.lock().await;
        model.generate_with_options(
            &req.prompt,
            request_generation_options(req.max_tokens, req.temperature, req.top_p),
        )?
    };

    let resp = make_completion_response(req.model, text);

    let tokens = resp
        .choices
        .first()
        .map_or(0, |c| c.text.split_whitespace().count());
    state
        .metrics
        .tokens_generated_total
        .with_label_values(&[&resp.model])
        .inc_by(tokens as u64);

    timer.observe_duration();
    state
        .metrics
        .inference_requests_total
        .with_label_values(&["completions", "200"])
        .inc();

    Ok(Json(resp))
}

/// Handle /v1/chat/completions
async fn handle_chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    validate_generation_options(req.stream, req.temperature, req.top_p)?;

    let timer = state
        .metrics
        .inference_duration_seconds
        .with_label_values(&["chat"])
        .start_timer();

    state.metrics.queue_depth.inc();
    let _permit = state.semaphore.clone().acquire_owned().await?;
    state.metrics.queue_depth.dec();

    let prompt = build_chat_prompt(&req.messages);
    let text = {
        let mut model = state.model.lock().await;
        model.generate_with_options(
            &prompt,
            request_generation_options(req.max_tokens, req.temperature, req.top_p),
        )?
    };

    let resp = make_chat_response(req.model, text);

    let tokens = resp
        .choices
        .first()
        .map_or(0, |c| c.message.content.split_whitespace().count());
    state
        .metrics
        .tokens_generated_total
        .with_label_values(&[&resp.model])
        .inc_by(tokens as u64);

    timer.observe_duration();
    state
        .metrics
        .inference_requests_total
        .with_label_values(&["chat", "200"])
        .inc();

    Ok(Json(resp))
}

/// Handle /v1/models
async fn handle_list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let model = state.model.try_lock();
    let model_id = model
        .ok()
        .and_then(|ctx| ctx.config().name.clone())
        .unwrap_or_else(|| "powerinfer-model".to_string());

    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "powerinfer",
                "permission": []
            }
        ]
    }))
}

/// Health check
async fn handle_health() -> &'static str {
    "OK"
}

/// Handle /metrics - Prometheus exposition format
async fn handle_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let uptime = chrono::Utc::now().timestamp() as u64 - state.start_epoch;
    state.metrics.uptime_seconds.set(uptime as i64);
    let body = state.metrics.gather();
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_token_budget_bounds() {
        assert_eq!(request_token_budget(None), 64);
        assert_eq!(request_token_budget(Some(0)), 1);
        assert_eq!(request_token_budget(Some(8)), 8);
        assert_eq!(request_token_budget(Some(600)), 512);
    }

    #[test]
    fn test_build_chat_prompt() {
        let prompt = build_chat_prompt(&[
            ChatMessage {
                role: "system".to_string(),
                content: "Be concise".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
        ]);

        assert_eq!(prompt, "system: Be concise\nuser: Hello");
    }

    #[test]
    fn test_validate_generation_options() {
        assert!(validate_generation_options(None, None, None).is_ok());
        assert!(validate_generation_options(Some(false), Some(0.0), Some(1.0)).is_ok());
        assert!(validate_generation_options(None, Some(0.7), Some(0.9)).is_ok());
        assert!(validate_generation_options(Some(true), None, None).is_err());
        assert!(validate_generation_options(None, Some(-0.1), None).is_err());
        assert!(validate_generation_options(None, None, Some(1.1)).is_err());
    }

    #[test]
    fn test_make_chat_response_shape() {
        let resp = make_chat_response("demo".to_string(), "Paris".to_string());

        assert_eq!(resp.object, "chat.completion");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.role, "assistant");
        assert_eq!(resp.choices[0].message.content, "Paris");
    }

    #[tokio::test]
    async fn test_server_types() {
        let _config = ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 8080,
            model_path: "test.gguf".to_string(),
            max_concurrent: 4,
            max_queue_depth: 64,
            hot_index_path: None,
        };
    }
}
