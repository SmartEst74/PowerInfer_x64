//! OpenAI-compatible HTTP API server
//!
//! Provides /v1/completions and /v1/chat/completions endpoints
//! using Axum and Tokio.

use axum::{
    routing::{post, get},
    Router, Json, extract::State, http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use tower_http::cors::{CorsLayer, Any};
use tracing::info;

use crate::InferenceContext;

/// Application state shared across requests
pub struct AppState {
    pub model: Arc<RwLock<Arc<InferenceContext>>>,
    pub max_concurrent: usize,
    pub semaphore: Arc<Semaphore>,
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_path: String,
    pub max_concurrent: usize,
    pub max_queue_depth: usize,
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

/// OpenAI-style request: Chat completions
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// API error type for axum handlers
pub struct ApiError(anyhow::Error);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "message": self.0.to_string(),
                    "type": "server_error"
                }
            })),
        )
            .into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

/// Build Axum router with all endpoints
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/completions", post(handle_completions))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/models", get(handle_list_models))
        .route("/health", get(handle_health))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .with_state(state)
}

/// Start the HTTP server
pub async fn serve(config: ServerConfig) -> anyhow::Result<()> {
    info!("Loading model from: {}", config.model_path);
    let ctx = InferenceContext::from_gguf(
        &config.model_path,
        crate::runtime::BackendFactory::cpu(),
    )?;
    info!(
        "Model loaded: {} ({} layers)",
        ctx.config().name.as_deref().unwrap_or("unknown"),
        ctx.config().block_count
    );

    let ctx = Arc::new(RwLock::new(Arc::new(ctx)));
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
    };

    let router = build_router(state);
    let addr = format!("{}:{}", config.host, config.port).parse()?;

    info!("PowerInfer_x64 server listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}

/// Handle /v1/completions
async fn handle_completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, ApiError> {
    let _permit = state.semaphore.clone().acquire_owned().await?;

    let _model = state.model.read().await;
    let text = format!("[DUMMY] Response to: {}", req.prompt);

    let resp = CompletionResponse {
        id: "cmpl-123".to_string(),
        object: "text_completion".to_string(),
        created: chrono::Utc::now().timestamp() as u64,
        model: req.model,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
    };

    Ok(Json(resp))
}

/// Handle /v1/chat/completions
async fn handle_chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<CompletionResponse>, ApiError> {
    let _permit = state.semaphore.clone().acquire_owned().await?;

    let prompt = req
        .messages
        .iter()
        .map(|m| format!("{}: {}\n", m.role, m.content))
        .collect::<String>();

    let _model = state.model.read().await;
    let text = format!("[DUMMY CHAT] Reply to: {}", prompt);

    let resp = CompletionResponse {
        id: "chatcmpl-123".to_string(),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp() as u64,
        model: req.model,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
    };

    Ok(Json(resp))
}

/// Handle /v1/models
async fn handle_list_models(
    State(_state): State<AppState>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "powerinfer-model",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_types() {
        let _config = ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 8080,
            model_path: "test.gguf".to_string(),
            max_concurrent: 4,
            max_queue_depth: 64,
        };
    }
}
