# Loci

A local LLM inference engine in Rust.

Loci loads GGUF-format language models on consumer hardware, providing both a CLI and an OpenAI-compatible HTTP API with support for streaming, tool calling, reasoning, and KV-cache prefix caching.

## Features

- **CLI and API server** — generate text from the terminal or via an HTTP API
- **OpenAI-compatible** — `/v1/chat/completions` with streaming, tools, logprobs, and `include_usage`
- **Tool/function calling** — structured output via XML, JSON, or Python-call formats
- **Reasoning support** — detects thinking tokens with configurable reasoning budgets
- **KV-cache prefix caching** — disk-based cache persistence for faster multi-turn conversations
- **Streaming** — token-by-token output both in CLI (`-s`) and SSE over HTTP
- **Quantized models** — Q4, Q8, and other GGUF quantization formats
- **CUDA acceleration** — optional GPU backend with flash attention

## Supported models

Currently verified with **LFM-2** (Liquid Foundation Models) architecture. The GGUF parsing and model infrastructure is designed to support additional architectures — adding a new model type requires implementing the `Model` trait and a config parser.

## Installation

```bash
cargo install loci
```

Or build from source:

```bash
git clone https://github.com/YOUR_USER/loci
cd loci
cargo build --release

# With CUDA support:
cargo build --release --features cuda
```

## Usage

### CLI

```bash
# Show model info
cargo run --release -- info models/model.gguf

# Generate text
cargo run --release -- generate "Once upon a time" models/model.gguf -m 100

# Chat with system message (streaming)
cargo run --release -- chat "Hello!" models/model.gguf -s -m 100
```

### API server

```bash
cargo run --release -- serve -b 127.0.0.1:8000 -c default_config.toml
```

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/LFM2.5-350M-Q8_0.gguf",
    "messages": [{"role": "user", "content": "Write a short story about Rust language"}],
    "max_tokens": 50,
    "temperature": 0.1,
    "stream": true,
    "stream_options": { "include_usage": true }
  }'
```

Response (streaming):

```
data: {"id":"8aae3771-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""}}],"usage":null}
data: {"id":"8aae3771-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Rust"}}],"usage":null}
data: {"id":"8aae3771-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" is"}}],"usage":null}
...
```

## Configuration

Loci uses a TOML configuration file with three sections:

| Section | Description |
|---------|-------------|
| `[inference]` | Compute dtype, max sequence length, flash attention |
| `[generation]` | Sampling parameters, reasoning effort, tool choice |
| `[cache]` | Prefix caching, cache directory, size limits |

See [`default_config.toml`](./default_config.toml) for all options. CLI flags override config file values.

## Project structure

```
src/
├── lib.rs               # Library root
├── main.rs              # Entry point
├── cli.rs               # CLI argument parsing and dispatch
├── error.rs             # Error types
├── types.rs             # Core types (messages, tools, etc.)
├── render.rs            # Streaming output rendering
├── profiling.rs         # Profiling macros
├── tokenizer.rs         # Tokenizer service
├── config/              # Configuration subsystem
│   ├── file_config.rs
│   ├── inference_config.rs
│   ├── generation_config.rs
│   ├── model_cache_config.rs
│   ├── model_config.rs
│   ├── tokenizer_config.rs
│   └── parser/          # Architecture-specific parsers
├── gguf/                # GGUF file parser
│   ├── types.rs
│   └── loader.rs
├── model/               # Model implementations
│   ├── model_base.rs    # Model trait
│   ├── model_impls/     # Concrete architectures
│   └── utility.rs
├── inference/           # Inference engine
│   ├── engine.rs
│   ├── sampler.rs
│   ├── generation_handler.rs
│   ├── generation_context.rs
│   ├── model_cache.rs
│   ├── reasoning_supervisor.rs
│   ├── tool_calling_supervisor.rs
│   ├── tool_formatter.rs
│   └── stop_pattern_matcher.rs
├── session/             # Chat session management
└── api/                 # HTTP API server
    ├── server.rs
    ├── handlers.rs
    ├── types.rs
    └── worker.rs
tests/
├── api_server.rs
├── config_file_loading.rs
├── pipeline_generation.rs
└── fixtures/
    └── mod.rs
```

## License

MIT
