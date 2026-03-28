//! PowerInfer_x64 HTTP server entry point

use powerinfer::server::{serve, ServerConfig};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "model.gguf".to_string());
    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse()
        .unwrap_or(8080);

    let config = ServerConfig {
        host: "0.0.0.0".to_string(),
        port,
        model_path,
        max_concurrent: 4,
        max_queue_depth: 64,
    };

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(serve(config))
}
