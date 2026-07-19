mod fixtures;

use candle_core::DType;
use loci::config::{
    FileConfig, GenerationConfig, InferenceConfig, ModelArchitecture, ModelCacheConfig,
    ModelConfig, TokenizerConfig,
};
use loci::gguf::Loader;
use loci::inference::ToolFormatStyle;
use loci::types::{ModelCacheFragmentation, ReasoningEffort, ToolChoice, ToolChoiceMode};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_config_file_loading() {
    let expected_inference_config = InferenceConfig {
        dtype: DType::F16,
        max_seq_len: 32_000,
        flash_attn: true,
        conv_on_cpu: true,
    };
    let expected_generation_config = GenerationConfig {
        stop_tokens: None,
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: 4096,
        repetition_penalty: 1.15,
        tool_choice: ToolChoice::Mode(ToolChoiceMode::Auto),
        reasoning_effort: ReasoningEffort::High,
        logprobs: false,
        top_logprobs: Some(5),
        seed: 19,
    };
    let expected_cache_config = ModelCacheConfig {
        prefix_caching: true,
        cache_dir: PathBuf::from("model_cache"),
        max_cache_size: 16_000_000_000,
        min_cache_tokens: 512,
        fragmentation: ModelCacheFragmentation::BlockWise { block_size: 32 },
    };

    let (_gguf_temp_dir, gguf_path) = fixtures::create_fake_lfm2_gguf();
    let mut expected_model_config = ModelConfig::default();
    expected_model_config.architecture = ModelArchitecture::Lfm2;
    expected_model_config.file_path = gguf_path.clone();
    expected_model_config.model_name = "test".to_string();
    expected_model_config.n_layers = 16;
    expected_model_config.max_seq_len = 4096;
    expected_model_config.hidden_size = 1024;
    expected_model_config.intermediate_ffn_size = 4608;
    expected_model_config.n_heads = 16;
    expected_model_config.vocab_size = 16;
    expected_model_config.n_kv_heads = vec![0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0];
    expected_model_config.conv_l_cache = Some(3);
    expected_model_config.rope_theta = 1000000.0;
    expected_model_config.rms_epsilon = 0.00001;
    expected_model_config.cache_seq_len_dim = 2;
    expected_model_config.supports_reasoning = false;
    expected_model_config.supports_tool_calling = true;
    expected_model_config.tool_call_start_token_id = Some(10);
    expected_model_config.tool_call_end_token_id = Some(11);
    expected_model_config.flatten_tools_to_functions = true;
    expected_model_config.tool_call_format_style = ToolFormatStyle::PythonCall;

    let mut expected_tokenizer_config = TokenizerConfig::default();
    expected_tokenizer_config.model_type = Some("gpt2".to_string());
    expected_tokenizer_config.chat_template = Some(r###"{{- bos_token -}}
{%- set keep_past_thinking = keep_past_thinking | default(false) -%}
{%- set ns = namespace(system_prompt="") -%}
{%- if messages[0]["role"] == "system" -%}
    {%- set sys_content = messages[0]["content"] -%}
    {%- if sys_content is not string -%}
        {%- for item in sys_content -%}
            {%- if item["type"] == "text" -%}
                {%- set ns.system_prompt = ns.system_prompt + item["text"] -%}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {%- set ns.system_prompt = sys_content -%}
    {%- endif -%}
    {%- set messages = messages[1:] -%}
{%- endif -%}
{%- if tools -%}
    {%- set ns.system_prompt = ns.system_prompt + ("\n" if ns.system_prompt else "") + "List of tools: [" -%}
    {%- for tool in tools -%}
        {%- if tool is not string -%}
            {%- set tool = tool | tojson -%}
        {%- endif -%}
        {%- set ns.system_prompt = ns.system_prompt + tool -%}
        {%- if not loop.last -%}
            {%- set ns.system_prompt = ns.system_prompt + ", " -%}
        {%- endif -%}
    {%- endfor -%}
    {%- set ns.system_prompt = ns.system_prompt + "]" -%}
{%- endif -%}
{%- if ns.system_prompt -%}
    {{- "<|im_start|>system\n" + ns.system_prompt + "<|im_end|>\n" -}}
{%- endif -%}
{%- set ns.last_assistant_index = -1 -%}
{%- for message in messages -%}
    {%- if message["role"] == "assistant" -%}
        {%- set ns.last_assistant_index = loop.index0 -%}
    {%- endif -%}
{%- endfor -%}
{%- for message in messages -%}
    {{- "<|im_start|>" + message["role"] + "\n" -}}
    {%- if message.get('tool_calls') %}
        {# ───── create a list to append tool calls to ───── #}
        {%- set tool_calls_ns = namespace(tool_calls=[])%}
        {%- for tool_call in message['tool_calls'] %}
            {%- set func_name = tool_call['function']['name'] %}
            {%- set func_args = tool_call['function']['arguments'] %}
            {# ───── create a list of func_arg strings to accumulate for each tool call ───── #}
            {%- set args_ns = namespace(arg_strings=[])%}
            {%- for arg_name, arg_value in func_args.items() %}
                {%- if arg_value is none %}
                    {%- set formatted_arg_value = 'null' %}
                {%- elif arg_value is boolean %}
                    {%- set formatted_arg_value = 'True' if arg_value else 'False' %}
                {%- elif arg_value is string %}
                    {%- set formatted_arg_value = '"' ~ arg_value ~ '"' %}
                {%- elif arg_value is mapping or arg_value is iterable %}
                    {%- set formatted_arg_value = arg_value | tojson %}
                {%- else %}
                    {%- set formatted_arg_value = arg_value | string %}
                {%- endif %}
                {# ───── format each argument key,value pair ───── #}
                {%- set args_ns.arg_strings =  args_ns.arg_strings + [arg_name ~ '=' ~ formatted_arg_value] %}
            {%- endfor %}
            {# ───── append each formatted tool call ───── #}
            {%- set tool_calls_ns.tool_calls = tool_calls_ns.tool_calls + [(func_name + '(' + (args_ns.arg_strings | join(", ")) + ')' )]%}
        {%- endfor %}
        {# ───── format the final tool calls ───── #}
        {{-'<|tool_call_start|>[' + (tool_calls_ns.tool_calls | join(", ")) + ']<|tool_call_end|>'}}
    {%- endif %}
    {%- set content = message["content"] -%}
    {%- if content is not string -%}
        {%- set ns.content = "" -%}
        {%- for item in content -%}
            {%- if item["type"] == "image" -%}
                {%- set ns.content = ns.content + "<image>" -%}
            {%- elif item["type"] == "text" -%}
                {%- set ns.content = ns.content + item["text"] -%}
            {%- else -%}
                {%- set ns.content = ns.content + item | tojson -%}
            {%- endfor -%}
        {%- endfor -%}
        {%- set content = ns.content -%}
    {%- endif -%}
    {%- if message["role"] == "assistant" and not keep_past_thinking and loop.index0 != ns.last_assistant_index -%}
        {%- if "</think>" in content -%}
            {%- set content = content.split("</think>")[-1] | trim -%}
        {%- endif -%}
    {%- endif -%}
    {{- content + "<|im_end|>\n" -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- "<|im_start|>assistant\n" -}}
{%- endif -%}"###.to_string());
    expected_tokenizer_config.bos_token_id = Some(1);
    expected_tokenizer_config.eos_token_id = Some(7);
    expected_tokenizer_config.padding_token_id = Some(0);
    expected_tokenizer_config.tokens = Some(vec![
        String::from("<|pad|>"),
        String::from("<|startoftext|>"),
        String::from("<|endoftext|>"),
        String::from("<|fim_pre|>"),
        String::from("<|fim_mid|>"),
        String::from("<|fim_suf|>"),
        String::from("<|im_start|>"),
        String::from("<|im_end|>"),
        String::from("<|tool_list_start|>"),
        String::from("<|tool_list_end|>"),
        String::from("<|tool_call_start|>"),
        String::from("<|tool_call_end|>"),
        String::from("<|tool_response_start|>"),
        String::from("<|tool_response_end|>"),
        String::from("<|reserved_4|>"),
        String::from("<|reserved_5|>"),
    ]);
    expected_tokenizer_config.token_type =
        Some(vec![3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1]);
    expected_tokenizer_config.merges = Some(vec![
        String::from("a b"),
        String::from("c d"),
        String::from("e f"),
        String::from("g h"),
        String::from("i j"),
        String::from("k l"),
        String::from("m n"),
        String::from("o p"),
        String::from("q r"),
        String::from("s t"),
        String::from("u v"),
        String::from("w x"),
        String::from("y z"),
        String::from("r j"),
        String::from("i b"),
        String::from("e s"),
    ]);
    expected_tokenizer_config.add_bos = true;
    expected_tokenizer_config.add_eos = false;

    let (_toml_temp_dir, toml_path) = fixtures::create_fake_toml_config();
    let FileConfig {
        generation_config,
        inference_config,
        cache_config,
    } = FileConfig::load(&toml_path)
        .expect(&format!("failed to load config from file: {:?}", toml_path));

    let inference_config = InferenceConfig::builder()
        .with_file_config(inference_config)
        .build();
    assert_eq!(inference_config, expected_inference_config);

    let generation_config = GenerationConfig::builder()
        .with_file_config(generation_config)
        .build();
    assert_eq!(generation_config, expected_generation_config);

    let cache_config = ModelCacheConfig::builder()
        .with_file_config(cache_config)
        .build();
    assert_eq!(cache_config, expected_cache_config);

    let gguf_info = Loader::load_gguf_info(&gguf_path, 0, false).expect("failed to load gguf info");
    let model_config =
        ModelConfig::from_gguf_info(&gguf_info).expect("failed to parse model config");
    assert_eq!(model_config, expected_model_config);

    let tokenizer_config = TokenizerConfig::from(gguf_info.kv_meta.as_slice());
    assert_eq!(tokenizer_config, expected_tokenizer_config);
}

#[test]
fn test_load_gguf_corrupt_magic() {
    let tmp_dir = TempDir::new().expect("temp dir failed");
    let file_path = tmp_dir.path().join("corrupt.gguf");
    let mut f = File::create(&file_path).unwrap();
    f.write_all(b"NOTG").unwrap(); // wrong magic
    let result = Loader::load_gguf_info(&file_path, 0, false);
    assert!(result.is_err(), "expected error for corrupt magic");
}

#[test]
fn test_load_gguf_nonexistent() {
    let result = Loader::load_gguf_info(PathBuf::from("/nonexistent/path.gguf"), 0, false);
    assert!(result.is_err(), "expected error for nonexistent file");
}

#[test]
fn test_load_toml_nonexistent() {
    let result = FileConfig::load(PathBuf::from("/nonexistent/config.toml"));
    assert!(result.is_err(), "expected error for nonexistent file");
}

#[test]
fn test_load_toml_invalid_content() {
    let tmp_dir = TempDir::new().expect("temp dir failed");
    let file_path = tmp_dir.path().join("bad.toml");
    let mut f = File::create(&file_path).unwrap();
    f.write_all(b"[[[invalid toml").unwrap();
    let result = FileConfig::load(&file_path);
    assert!(result.is_err(), "expected error for invalid toml");
}
