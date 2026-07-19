use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use tempfile::TempDir;

pub fn create_fake_lfm2_gguf() -> (TempDir, PathBuf) {
    let tmp_dir = TempDir::new().expect("failed to create tempdir");
    let file_path = tmp_dir.path().join("test.gguf");

    let mut fake_gguf = vec![];
    // Magic
    fake_gguf.extend_from_slice(b"GGUF");
    // Version
    fake_gguf.extend_from_slice(&3u32.to_le_bytes());
    // Tensor Count
    fake_gguf.extend_from_slice(&0u64.to_le_bytes());
    // Metadata KV Count
    fake_gguf.extend_from_slice(&23u64.to_le_bytes());

    // Metadata KV
    // Len string
    fake_gguf.extend_from_slice(&("general.alignment".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"general.alignment");
    // Gguf type
    fake_gguf.extend_from_slice(&4i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&32u32.to_le_bytes());

    // Len string
    fake_gguf.extend_from_slice(&("general.architecture".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"general.architecture");
    // Gguf type
    fake_gguf.extend_from_slice(&8i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&("lfm2".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"lfm2");

    // Len string
    fake_gguf.extend_from_slice(&("general.name".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"general.name");
    // Gguf type
    fake_gguf.extend_from_slice(&8i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&("test".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"test");

    // Len string
    fake_gguf.extend_from_slice(&("tokenizer.ggml.model".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"tokenizer.ggml.model");
    // Gguf type
    fake_gguf.extend_from_slice(&8i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&("gpt2".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"gpt2");

    // Len string
    fake_gguf.extend_from_slice(&("tokenizer.chat_template".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"tokenizer.chat_template");
    // Gguf type
    fake_gguf.extend_from_slice(&8i32.to_le_bytes());
    // Value
    let chat_template = r###"{{- bos_token -}}
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
{%- endif -%}"###.to_string();
    fake_gguf.extend_from_slice(&(chat_template.len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(chat_template.as_bytes());

    // Len string
    fake_gguf.extend_from_slice(&("lfm2.block_count".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"lfm2.block_count");
    // Gguf type
    fake_gguf.extend_from_slice(&4i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&16u32.to_le_bytes());

    // Len string
    fake_gguf.extend_from_slice(&("lfm2.context_length".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"lfm2.context_length");
    // Gguf type
    fake_gguf.extend_from_slice(&4i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&4096u32.to_le_bytes());

    // Len string
    fake_gguf.extend_from_slice(&("lfm2.embedding_length".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"lfm2.embedding_length");
    // Gguf type
    fake_gguf.extend_from_slice(&4i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&1024u32.to_le_bytes());

    // Len string
    fake_gguf.extend_from_slice(&("lfm2.feed_forward_length".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"lfm2.feed_forward_length");
    // Gguf type
    fake_gguf.extend_from_slice(&4i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&4608u32.to_le_bytes());

    // Len string
    fake_gguf.extend_from_slice(&("lfm2.attention.head_count".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"lfm2.attention.head_count");
    // Gguf type
    fake_gguf.extend_from_slice(&4i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&16u32.to_le_bytes());

    // Len string
    fake_gguf.extend_from_slice(&("lfm2.vocab_size".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"lfm2.vocab_size");
    // Gguf type
    fake_gguf.extend_from_slice(&4i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&16u32.to_le_bytes());

    // Len string
    fake_gguf.extend_from_slice(&("lfm2.shortconv.l_cache".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"lfm2.shortconv.l_cache");
    // Gguf type
    fake_gguf.extend_from_slice(&4i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&3u32.to_le_bytes());

    // Len string
    fake_gguf.extend_from_slice(&("tokenizer.ggml.bos_token_id".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"tokenizer.ggml.bos_token_id");
    // Gguf type
    fake_gguf.extend_from_slice(&4i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&1u32.to_le_bytes());

    // Len string
    fake_gguf.extend_from_slice(&("tokenizer.ggml.eos_token_id".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"tokenizer.ggml.eos_token_id");
    // Gguf type
    fake_gguf.extend_from_slice(&4i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&7u32.to_le_bytes());

    // Len string
    fake_gguf.extend_from_slice(&("tokenizer.ggml.padding_token_id".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"tokenizer.ggml.padding_token_id");
    // Gguf type
    fake_gguf.extend_from_slice(&4i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&0u32.to_le_bytes());

    // Len string
    fake_gguf.extend_from_slice(&("lfm2.attention.head_count_kv".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"lfm2.attention.head_count_kv");
    // Gguf type
    fake_gguf.extend_from_slice(&9i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&4i32.to_le_bytes());
    fake_gguf.extend_from_slice(&16i64.to_le_bytes());
    fake_gguf.extend_from_slice(&0u32.to_le_bytes());
    fake_gguf.extend_from_slice(&0u32.to_le_bytes());
    fake_gguf.extend_from_slice(&8u32.to_le_bytes());
    fake_gguf.extend_from_slice(&0u32.to_le_bytes());
    fake_gguf.extend_from_slice(&0u32.to_le_bytes());
    fake_gguf.extend_from_slice(&8u32.to_le_bytes());
    fake_gguf.extend_from_slice(&0u32.to_le_bytes());
    fake_gguf.extend_from_slice(&0u32.to_le_bytes());
    fake_gguf.extend_from_slice(&8u32.to_le_bytes());
    fake_gguf.extend_from_slice(&0u32.to_le_bytes());
    fake_gguf.extend_from_slice(&8u32.to_le_bytes());
    fake_gguf.extend_from_slice(&0u32.to_le_bytes());
    fake_gguf.extend_from_slice(&8u32.to_le_bytes());
    fake_gguf.extend_from_slice(&0u32.to_le_bytes());
    fake_gguf.extend_from_slice(&8u32.to_le_bytes());
    fake_gguf.extend_from_slice(&0u32.to_le_bytes());

    // Len string
    fake_gguf.extend_from_slice(&("tokenizer.ggml.tokens".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"tokenizer.ggml.tokens");
    // Gguf type
    fake_gguf.extend_from_slice(&9i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&8i32.to_le_bytes());
    fake_gguf.extend_from_slice(&16i64.to_le_bytes());
    fake_gguf.extend_from_slice(&("<|pad|>".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"<|pad|>");
    fake_gguf.extend_from_slice(&("<|startoftext|>".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"<|startoftext|>");
    fake_gguf.extend_from_slice(&("<|endoftext|>".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"<|endoftext|>");
    fake_gguf.extend_from_slice(&("<|fim_pre|>".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"<|fim_pre|>");
    fake_gguf.extend_from_slice(&("<|fim_mid|>".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"<|fim_mid|>");
    fake_gguf.extend_from_slice(&("<|fim_suf|>".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"<|fim_suf|>");
    fake_gguf.extend_from_slice(&("<|im_start|>".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"<|im_start|>");
    fake_gguf.extend_from_slice(&("<|im_end|>".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"<|im_end|>");
    fake_gguf.extend_from_slice(&("<|tool_list_start|>".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"<|tool_list_start|>");
    fake_gguf.extend_from_slice(&("<|tool_list_end|>".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"<|tool_list_end|>");
    fake_gguf.extend_from_slice(&("<|tool_call_start|>".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"<|tool_call_start|>");
    fake_gguf.extend_from_slice(&("<|tool_call_end|>".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"<|tool_call_end|>");
    fake_gguf.extend_from_slice(&("<|tool_response_start|>".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"<|tool_response_start|>");
    fake_gguf.extend_from_slice(&("<|tool_response_end|>".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"<|tool_response_end|>");
    fake_gguf.extend_from_slice(&("<|reserved_4|>".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"<|reserved_4|>");
    fake_gguf.extend_from_slice(&("<|reserved_5|>".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"<|reserved_5|>");

    // Len string
    fake_gguf.extend_from_slice(&("tokenizer.ggml.token_type".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"tokenizer.ggml.token_type");
    // Gguf type
    fake_gguf.extend_from_slice(&9i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&5i32.to_le_bytes());
    fake_gguf.extend_from_slice(&16i64.to_le_bytes());
    fake_gguf.extend_from_slice(&3i32.to_le_bytes());
    fake_gguf.extend_from_slice(&3i32.to_le_bytes());
    fake_gguf.extend_from_slice(&3i32.to_le_bytes());
    fake_gguf.extend_from_slice(&3i32.to_le_bytes());
    fake_gguf.extend_from_slice(&3i32.to_le_bytes());
    fake_gguf.extend_from_slice(&3i32.to_le_bytes());
    fake_gguf.extend_from_slice(&3i32.to_le_bytes());
    fake_gguf.extend_from_slice(&3i32.to_le_bytes());
    fake_gguf.extend_from_slice(&3i32.to_le_bytes());
    fake_gguf.extend_from_slice(&3i32.to_le_bytes());
    fake_gguf.extend_from_slice(&3i32.to_le_bytes());
    fake_gguf.extend_from_slice(&3i32.to_le_bytes());
    fake_gguf.extend_from_slice(&3i32.to_le_bytes());
    fake_gguf.extend_from_slice(&3i32.to_le_bytes());
    fake_gguf.extend_from_slice(&1i32.to_le_bytes());
    fake_gguf.extend_from_slice(&1i32.to_le_bytes());

    // Len string
    fake_gguf.extend_from_slice(&("tokenizer.ggml.merges".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"tokenizer.ggml.merges");
    // Gguf type
    fake_gguf.extend_from_slice(&9i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&8i32.to_le_bytes());
    fake_gguf.extend_from_slice(&16i64.to_le_bytes());
    fake_gguf.extend_from_slice(&("a b".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"a b");
    fake_gguf.extend_from_slice(&("c d".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"c d");
    fake_gguf.extend_from_slice(&("e f".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"e f");
    fake_gguf.extend_from_slice(&("g h".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"g h");
    fake_gguf.extend_from_slice(&("i j".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"i j");
    fake_gguf.extend_from_slice(&("k l".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"k l");
    fake_gguf.extend_from_slice(&("m n".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"m n");
    fake_gguf.extend_from_slice(&("o p".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"o p");
    fake_gguf.extend_from_slice(&("q r".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"q r");
    fake_gguf.extend_from_slice(&("s t".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"s t");
    fake_gguf.extend_from_slice(&("u v".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"u v");
    fake_gguf.extend_from_slice(&("w x".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"w x");
    fake_gguf.extend_from_slice(&("y z".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"y z");
    fake_gguf.extend_from_slice(&("r j".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"r j");
    fake_gguf.extend_from_slice(&("i b".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"i b");
    fake_gguf.extend_from_slice(&("e s".len() as u64).to_le_bytes());
    fake_gguf.extend_from_slice(b"e s");

    // Len string
    fake_gguf.extend_from_slice(&("lfm2.rope.freq_base".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"lfm2.rope.freq_base");
    // Gguf type
    fake_gguf.extend_from_slice(&6i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&1000000f32.to_le_bytes());

    // Len string
    fake_gguf
        .extend_from_slice(&("lfm2.attention.layer_norm_rms_epsilon".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"lfm2.attention.layer_norm_rms_epsilon");
    // Gguf type
    fake_gguf.extend_from_slice(&6i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&0.00001f32.to_le_bytes());

    // Len string
    fake_gguf.extend_from_slice(&("tokenizer.ggml.add_bos_token".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"tokenizer.ggml.add_bos_token");
    // Gguf type
    fake_gguf.extend_from_slice(&7i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&1i8.to_le_bytes());

    // Len string
    fake_gguf.extend_from_slice(&("tokenizer.ggml.add_eos_token".len() as u64).to_le_bytes());
    // String
    fake_gguf.extend_from_slice(b"tokenizer.ggml.add_eos_token");
    // Gguf type
    fake_gguf.extend_from_slice(&7i32.to_le_bytes());
    // Value
    fake_gguf.extend_from_slice(&0i8.to_le_bytes());

    let mut file = File::create(&file_path).expect("failed to create temp gguf file");
    file.write_all(&fake_gguf)
        .expect("failed to write to temp gguf file");

    (tmp_dir, file_path)
}

pub fn create_fake_toml_config() -> (TempDir, PathBuf) {
    let tmp_dir = TempDir::new().expect("temp dir failed");
    let file_path = tmp_dir.path().join("test_config.toml");

    let config = toml::toml! {
        [inference]
        dtype = "f16"
        max_seq_len = 32000
        conv_on_cpu = true
        flash_attn = true

        [generation]
        system_message = "You are a helpfull assistant."
        temperature = 0.7
        top_p = 0.9
        max_tokens = 4096
        repetition_penalty = 1.15
        tool_choice = "auto"
        reasoning_effort = "high"
        logprobs = false
        top_logprobs = 5
        seed = 19

        [cache]
        prefix_caching = true
        cache_dir = "model_cache"
        max_cache_size = 16_000_000_000_u64
        min_cache_tokens = 512
        fragmentation = "BlockWise(32)"
    };
    let toml_string = toml::to_string(&config).expect("failed to convert toml to string");

    let mut file = File::create(&file_path).expect("failed to create temp toml file");
    file.write_all(toml_string.as_bytes())
        .expect("failed to write to temp toml file");

    (tmp_dir, file_path)
}
