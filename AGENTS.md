# Loci — Agent guide

Loci is a Rust local LLM inference engine. It loads GGUF models and runs them via CLI or an OpenAI-compatible HTTP API.

## Build

```bash
cargo build                    # debug
cargo build --release          # release
cargo build --features cuda    # with CUDA
cargo check-errors             # check without warnings (alias)
```

## Run

```bash
# CLI generation
cargo run --release -- generate "prompt" path/to/model.gguf -m 100

# Chat (streaming)
cargo run --release -- chat "Hello" path/to/model.gguf -s -m 100

# API server
cargo run --release -- serve -b 127.0.0.1:8000 -c default_config.toml
```

## Key conventions

- **Edition 2024**, stable Rust.
- **Builder pattern** for configs (`InferenceConfig::builder().dtype(...).build()`).
- **thiserror** for error enums; `anyhow` for top-level error propagation.
- **tracing** for logging (not `log`/`println`). Set `RUST_LOG=debug` for verbose output.
- **Run tests with** `cargo test --features mock`.
- mockall is an optional dependency; integration tests need `mock` feature for `MockModel`/`MockTokenizer`.
- Integration tests in `tests/`: `pipeline_generation.rs`, `config_file_loading.rs`, `api_server.rs`.
- Shared test fixtures in `tests/fixtures/mod.rs`.
- **GGUF parsing** is in `src/gguf/`; model architectures in `src/model/model_impls/`.
- To add a new model architecture: implement `Model` trait in `model_impls/` + config parser in `config/parser/`.

## Architecture notes

- `InferenceEngine` orchestrates tokenization → model forward → sampling → streaming.
- `GenerationHandler` is the state machine for reasoning/tool-calling/content transitions.
- `EngineWorker` manages model lifecycle (load/unload on idle timeout) in server mode.
- Prefix cache lives on disk under `model_cache/` as `.safetensors` files.

## Config

See `default_config.toml`. CLI flags override file config. Sections: `[inference]`, `[generation]`, `[cache]`.
