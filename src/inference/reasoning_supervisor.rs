use crate::inference::{GenerationDataType, GenerationEvent, PostSamplingConfig};
use crate::types::ReasoningEffort;
#[cfg(test)]
use mockall::automock;

const HIGH_REASONING_BUDGET: u32 = 16384;
const MEDIUM_REASONING_BUDGET: u32 = 4096;
const LOW_REASONING_BUDGET: u32 = 1024;

#[cfg_attr(test, automock)]
pub trait ReasoningSupervisorInterface {
    fn advance(
        &mut self,
        token_ids: &[u32],
        ongoing_gen_type: &GenerationDataType,
    ) -> GenerationEvent;

    fn reasoning_tokens_count(&self) -> u32;
    fn get_reasoning_budget(&self) -> u32;
}

#[derive(Default, PartialEq, Debug)]
pub struct ReasoningSupervisor {
    pub reasoning_budget: u32,
    pub reasoning_tokens: u32,
    pub reasoning_start_token_id: u32,
    pub reasoning_end_token_id: u32,
}

impl ReasoningSupervisorInterface for ReasoningSupervisor {
    fn advance(
        &mut self,
        token_ids: &[u32],
        ongoing_gen_type: &GenerationDataType,
    ) -> GenerationEvent {
        if ongoing_gen_type == &GenerationDataType::Reasoning {
            self.reasoning_tokens += token_ids.len() as u32;
            if self.detect_reasoning_end(token_ids) {
                GenerationEvent::ReasoningStopped
            } else if self.reasoning_budget_exceeded() {
                GenerationEvent::ForceTokens {
                    tokens: vec![self.reasoning_end_token_id],
                }
            } else {
                GenerationEvent::None
            }
        } else if self.detect_reasoning_start(token_ids) {
            GenerationEvent::ReasoningStarted
        } else {
            GenerationEvent::None
        }
    }

    fn reasoning_tokens_count(&self) -> u32 {
        self.reasoning_tokens.saturating_sub(1)
    }

    fn get_reasoning_budget(&self) -> u32 {
        self.reasoning_budget
    }
}

impl ReasoningSupervisor {
    pub fn new(
        supports_reasoning: bool,
        reasoning_effort: &ReasoningEffort,
        config: &PostSamplingConfig,
    ) -> Option<Self> {
        if !supports_reasoning || reasoning_effort == &ReasoningEffort::None {
            return None;
        }
        let reasoning_start_token_id = config.reasoning_start_token_id?;
        let reasoning_end_token_id = config.reasoning_end_token_id?;

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
        })
    }

    pub fn detect_reasoning_start(&self, token_ids: &[u32]) -> bool {
        token_ids.ends_with(std::slice::from_ref(&self.reasoning_start_token_id))
    }

    pub fn detect_reasoning_end(&self, token_ids: &[u32]) -> bool {
        token_ids.ends_with(std::slice::from_ref(&self.reasoning_end_token_id))
    }

    pub fn reasoning_budget_exceeded(&self) -> bool {
        self.reasoning_budget
            .saturating_sub(self.reasoning_tokens_count())
            == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(false, ReasoningEffort::None, PostSamplingConfig::default(), None)]
    #[case(true, ReasoningEffort::None, PostSamplingConfig::default(), None)]
    #[case(true, ReasoningEffort::Low, PostSamplingConfig {
        reasoning_start_token_id: None,
        reasoning_end_token_id: Some(2),
        ..Default::default()
    }, None)]
    #[case(true, ReasoningEffort::Low, PostSamplingConfig {
        reasoning_start_token_id: Some(1),
        reasoning_end_token_id: None,
        ..Default::default()
    }, None)]
    #[case(true, ReasoningEffort::Low, PostSamplingConfig {
        reasoning_start_token_id: Some(1),
        reasoning_end_token_id: Some(2),
        ..Default::default()
    }, Some(ReasoningSupervisor {
        reasoning_budget: LOW_REASONING_BUDGET,
        reasoning_tokens: 0,
        reasoning_start_token_id: 1,
        reasoning_end_token_id: 2,
    }))]
    #[case(true, ReasoningEffort::Medium, PostSamplingConfig {
        reasoning_start_token_id: Some(1),
        reasoning_end_token_id: Some(2),
        ..Default::default()
    }, Some(ReasoningSupervisor {
        reasoning_budget: MEDIUM_REASONING_BUDGET,
        reasoning_tokens: 0,
        reasoning_start_token_id: 1,
        reasoning_end_token_id: 2,
    }))]
    #[case(true, ReasoningEffort::High, PostSamplingConfig {
        reasoning_start_token_id: Some(1),
        reasoning_end_token_id: Some(2),
        ..Default::default()
    }, Some(ReasoningSupervisor {
        reasoning_budget: HIGH_REASONING_BUDGET,
        reasoning_tokens: 0,
        reasoning_start_token_id: 1,
        reasoning_end_token_id: 2,
    }))]
    fn test_reasoning_supervisor_init(
        #[case] supports_reasoning: bool,
        #[case] reasoning_effort: ReasoningEffort,
        #[case] config: PostSamplingConfig,
        #[case] expected: Option<ReasoningSupervisor>,
    ) {
        assert_eq!(
            ReasoningSupervisor::new(supports_reasoning, &reasoning_effort, &config),
            expected
        );
    }

    #[rstest]
    #[case(vec![], false)]
    #[case(vec![1], true)]
    #[case(vec![3, 2, 3, 4, 1], true)]
    fn test_detect_reasoning_start(#[case] token_ids: Vec<u32>, #[case] expected: bool) {
        let reasoning_supervisor = ReasoningSupervisor {
            reasoning_start_token_id: 1,
            reasoning_end_token_id: 2,
            ..Default::default()
        };
        assert_eq!(
            reasoning_supervisor.detect_reasoning_start(&token_ids),
            expected
        );
    }

    #[rstest]
    #[case(vec![], false)]
    #[case(vec![2], true)]
    #[case(vec![3, 2, 3, 4, 2], true)]
    fn test_detect_reasoning_end(#[case] token_ids: Vec<u32>, #[case] expected: bool) {
        let reasoning_supervisor = ReasoningSupervisor {
            reasoning_start_token_id: 1,
            reasoning_end_token_id: 2,
            ..Default::default()
        };
        assert_eq!(
            reasoning_supervisor.detect_reasoning_end(&token_ids),
            expected
        );
    }

    #[rstest]
    #[case(0, 0, true)]
    #[case(0, 1, false)]
    #[case(1, 0, true)]
    #[case(1, 1, false)]
    #[case(1, 2, false)]
    #[case(2, 0, true)]
    #[case(2, 1, true)]
    #[case(2, 2, false)]
    fn test_reasoning_budget_exceeded(
        #[case] reasoning_tokens: u32,
        #[case] reasoning_budget: u32,
        #[case] expected: bool,
    ) {
        let reasoning_supervisor = ReasoningSupervisor {
            reasoning_budget,
            reasoning_tokens,
            ..Default::default()
        };
        assert_eq!(reasoning_supervisor.reasoning_budget_exceeded(), expected);
    }

    #[rstest]
    #[case(0, 0)]
    #[case(1, 0)]
    #[case(2, 1)]
    #[case(3, 2)]
    fn test_reasoning_tokens_count(#[case] reasoning_tokens: u32, #[case] expected: u32) {
        let reasoning_supervisor = ReasoningSupervisor {
            reasoning_tokens,
            ..Default::default()
        };
        assert_eq!(reasoning_supervisor.reasoning_tokens_count(), expected);
    }

    #[rstest]
    #[case(vec![1], GenerationDataType::DirectContent, 5, 0, GenerationEvent::ReasoningStarted)]
    #[case(vec![2], GenerationDataType::Reasoning, 0, 1, GenerationEvent::ReasoningStopped)]
    #[case(vec![3], GenerationDataType::Reasoning, 3, 3, GenerationEvent::ForceTokens { tokens: vec![2] })] // reasoning_tokens becomes 4 after advance increments it
    #[case(vec![4, 5, 6], GenerationDataType::Reasoning, 5, 0, GenerationEvent::None)]
    #[case(vec![7, 8, 9], GenerationDataType::DirectContent, 0, 0, GenerationEvent::None)]
    fn test_advance(
        #[case] token_ids: Vec<u32>,
        #[case] ongoing_gen_type: GenerationDataType,
        #[case] reasoning_budget: u32,
        #[case] reasoning_tokens: u32,
        #[case] expected: GenerationEvent,
    ) {
        let mut reasoning_supervisor = ReasoningSupervisor {
            reasoning_start_token_id: 1,
            reasoning_end_token_id: 2,
            reasoning_budget,
            reasoning_tokens,
        };
        assert_eq!(
            reasoning_supervisor.advance(&token_ids, &ongoing_gen_type),
            expected
        );
    }
}
