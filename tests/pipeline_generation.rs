mod fixtures;

use candle_core::{Device, Tensor};
use candle_nn::kv_cache::ConcatKvCache;
use fixtures::create_fake_lfm2_gguf;
use loci::config::GenerationOverrides;
use loci::inference::{GenerationContext, InferenceEngine, StreamCallback};
use loci::model::model_base::{MixedCache, MockModel, ModelCacheInfo, ModelCacheType};
use loci::tokenizer::MockTokenizer;
use loci::types::{FinishReason, Role};
use mockall::predicate::{always, eq};

#[test]
fn test_engine_with_mock_model() {
    let (_tmp_dir, gguf_path) = create_fake_lfm2_gguf();
    let input_tokens = vec![1u32, 2, 3];

    let mut mock_tokenizer = MockTokenizer::new();
    mock_tokenizer
        .expect_encode()
        .with(eq("hello"), eq(true))
        .returning(move |_, _| Ok(input_tokens.clone()));
    mock_tokenizer
        .expect_special_token_ids()
        .return_const(vec![1u32, 7]);
    mock_tokenizer.expect_eos_token_id().return_const(7u32);
    mock_tokenizer
        .expect_process_token_stream()
        .returning(move |_, _| Ok(Some("World".to_string())));

    let mut mock_model = MockModel::new();
    mock_model.expect_cache_info().returning(|| ModelCacheInfo {
        cache_type: ModelCacheType::MixedWithConv { conv_l_cache: 2 },
        cache_seq_len_dim: 2,
        n_layers: 16,
        cache_block_size_hint: 1,
    });
    mock_model.expect_init_cache().returning(|| {
        Ok(vec![
            None,
            None,
            Some(MixedCache::KvCache(ConcatKvCache::new(2))),
            None,
            None,
            Some(MixedCache::KvCache(ConcatKvCache::new(2))),
            None,
            None,
            Some(MixedCache::KvCache(ConcatKvCache::new(2))),
            None,
            Some(MixedCache::KvCache(ConcatKvCache::new(2))),
            None,
            Some(MixedCache::KvCache(ConcatKvCache::new(2))),
            None,
            Some(MixedCache::KvCache(ConcatKvCache::new(2))),
            None,
        ])
    });
    mock_model.expect_min_prefill_tokens().return_const(1usize);
    mock_model.expect_conv_on_cpu().return_const(true);
    mock_model
        .expect_forward()
        .with(always(), always(), eq(0), eq(true))
        .returning(move |_, _, _, _| {
            Ok(Tensor::new(
                vec![vec![vec![
                    0.01f32, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                    0.01, 0.01, 0.01, 0.01,
                ]]],
                &Device::Cpu,
            )
            .expect("first model forward output tensor creation failed"))
        });
    mock_model
        .expect_forward()
        .with(always(), always(), eq(3), eq(true))
        .returning(move |_, _, _, _| {
            Ok(Tensor::new(
                vec![vec![vec![
                    0.01f32, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01,
                    0.01, 0.01, 0.01, 0.01,
                ]]],
                &Device::Cpu,
            )
            .expect("second model forward output tensor creation failed"))
        });

    let engine = InferenceEngine::builder()
        .with_gguf_metadata(gguf_path)
        .with_tokenizer(Box::new(mock_tokenizer))
        .with_model(Box::new(mock_model))
        .build()
        .expect("failed to build inference engine");
    let mut gen_ctx = GenerationContext::new("test", None, engine.model_cache_info())
        .expect("failed to create generation context");
    let mut overrides = GenerationOverrides::default();
    overrides.temperature = Some(0.0);
    let echo_callback: StreamCallback = Box::new(|frame| {
        eprintln!("Stream callback recieved generation report: {:#?}", frame);
        Ok(())
    });
    let report = engine
        .generate_stream("hello", &mut gen_ctx, overrides, echo_callback)
        .expect("failed to generate stream");
    assert_eq!(report.chat_message.role, Role::Assistant);
    assert_eq!(report.chat_message.content, Some(String::from("World")));
    assert_eq!(report.finish_reason, FinishReason::Stop);
    assert_eq!(report.usage.prompt_tokens, 3);
    assert_eq!(report.usage.completion_tokens, 2);
    assert_eq!(report.usage.total_tokens, 5);
}
