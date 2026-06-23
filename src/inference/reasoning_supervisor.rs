use crate::types::ReasoningEffort;
use crate::inference::{PostSamplingConfig, GenerationDataType, GenerationEvent};

const HIGH_REASONING_BUDGET: u32 = 16384;
const MEDIUM_REASONING_BUDGET: u32 = 4096;
const LOW_REASONING_BUDGET: u32 = 1024;

#[derive(Default)]
pub struct ReasoningSupervisor {
    pub reasoning_budget: u32,
    pub reasoning_tokens: u32,
    pub reasoning_start_token_id: u32,
    pub reasoning_end_token_id: u32,
    pub reasoning_start_detected: bool,
    pub reasoning_end_detected: bool,
}

impl ReasoningSupervisor {
    pub fn new(supports_reasoning: bool, reasoning_effort: &ReasoningEffort, config: &PostSamplingConfig) -> Option<Self> {
        if !supports_reasoning ||reasoning_effort == &ReasoningEffort::None {
            return None;
        }
        let reasoning_start_token_id = config.reasoning_start_token_id?.clone();
        let reasoning_end_token_id = config.reasoning_end_token_id?.clone();
        
        Some(Self {
            reasoning_budget: match reasoning_effort {
                ReasoningEffort::None => 0,
                ReasoningEffort::Low => LOW_REASONING_BUDGET,
                ReasoningEffort::Medium => MEDIUM_REASONING_BUDGET,
                ReasoningEffort::High => HIGH_REASONING_BUDGET,
            },
            reasoning_tokens: 0,
            reasoning_start_token_id,
            reasoning_end_token_id,
            ..Default::default()
        })
    }

    pub fn reasoning_tokens_count(&self) -> u32 {
        self.reasoning_tokens.saturating_sub(1)
    }

    pub fn detect_reasoning_start(&self, token_ids: &[u32]) -> bool {
        token_ids.ends_with(std::slice::from_ref(&self.reasoning_start_token_id))
    }

    pub fn detect_reasoning_end(&self, token_ids: &[u32]) -> bool {
        token_ids.ends_with(std::slice::from_ref(&self.reasoning_end_token_id))
    }

    pub fn advance(&mut self, token_ids: &[u32], ongoing_gen_type: &GenerationDataType) -> GenerationEvent {
        if ongoing_gen_type == &GenerationDataType::Reasoning { 
            self.reasoning_tokens += token_ids.len() as u32;
            if self.detect_reasoning_end(token_ids) {
                GenerationEvent::ReasoningStopped
            } else if self.reasoning_budget_exceeded(){
                GenerationEvent::ForceTokens { tokens: vec![self.reasoning_end_token_id] }
            } else {
                GenerationEvent::None
            }
        } else if self.detect_reasoning_start(token_ids) {
            GenerationEvent::ReasoningStarted
        } else {
            GenerationEvent::None
        }
    }

    pub fn reasoning_budget_exceeded(&self) -> bool {
        self.reasoning_budget.saturating_sub(self.reasoning_tokens_count()) <= 0
    }
}