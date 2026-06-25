use crate::inference::{
    GenerationDataType, GenerationEvent, ReasoningSupervisor, Sampler, SamplingResult,
    ToolCallingSupervisor,
};
use crate::types::ToolCall;

use candle_core::Tensor;

pub struct GenerationHandler<'a> {
    pub input_tokens: Vec<u32>,
    pub sampler: Box<dyn Sampler>,
    pub reasoning_supervisor: Option<ReasoningSupervisor>,
    pub tool_calling_supervisor: Option<ToolCallingSupervisor<'a>>,
    pub ongoing_gen_type: GenerationDataType,
    pub tokens_to_force: Vec<u32>,
    pub eos_token_id: u32,
    pub tool_choice_template: Option<Vec<u32>>,
}

impl<'a> GenerationHandler<'a> {
    pub fn new(
        input_tokens: &[u32],
        sampler: Box<dyn Sampler>,
        mut reasoning_supervisor: Option<ReasoningSupervisor>,
        mut tool_calling_supervisor: Option<ToolCallingSupervisor<'a>>,
        eos_token_id: u32,
        mut tool_choice_template: Option<Vec<u32>>,
    ) -> anyhow::Result<Self> {
        let mut ongoing_gen_type = GenerationDataType::DirectContent;
        let mut tokens_to_force = vec![];
        if let Some(reasoning_supervisor) = reasoning_supervisor.as_mut() {
            match reasoning_supervisor.advance(input_tokens, &ongoing_gen_type) {
                GenerationEvent::ReasoningStarted => {
                    ongoing_gen_type = GenerationDataType::Reasoning;
                }
                GenerationEvent::ReasoningStopped => {
                    ongoing_gen_type = GenerationDataType::DirectContent;
                }
                GenerationEvent::ForceTokens { tokens } => {
                    tokens_to_force.extend(tokens);
                }
                _ => {}
            }
        }

        if let Some(tool_calling_supervisor) = tool_calling_supervisor.as_mut()
            && let GenerationEvent::ToolCallStarted =
                tool_calling_supervisor.advance(input_tokens, &ongoing_gen_type)?
        {
            ongoing_gen_type = GenerationDataType::ToolCallName;
        }

        if ongoing_gen_type == GenerationDataType::DirectContent
            && let Some(template) = tool_choice_template.take()
        {
            tokens_to_force.extend(template);
        }

        Ok(Self {
            input_tokens: input_tokens.to_vec(),
            sampler,
            reasoning_supervisor,
            tool_calling_supervisor,
            ongoing_gen_type,
            tokens_to_force,
            eos_token_id,
            tool_choice_template,
        })
    }

    pub fn reasoning_token_count(&self) -> u32 {
        if let Some(ref reasoning_supervisor) = self.reasoning_supervisor {
            reasoning_supervisor.reasoning_tokens_count()
        } else {
            0
        }
    }

    pub fn set_input_tokens(&mut self, input_tokens: &[u32]) {
        self.input_tokens = input_tokens.to_vec();
    }

    pub fn advance(
        &mut self,
        logits: &Tensor,
        with_logprobs: bool,
        top_k_logprobs: Option<usize>,
    ) -> anyhow::Result<GenerationEvent> {
        let sampling_result = if self.tokens_to_force.is_empty() {
            if with_logprobs {
                self.sampler
                    .sample(logits, Some(top_k_logprobs.unwrap_or(0)))?
            } else {
                self.sampler.sample(logits, None)?
            }
        } else {
            SamplingResult {
                token: self.tokens_to_force.remove(0),
                logprob: None,
                top_k_logprobs: None,
            }
        };

        if sampling_result.token == self.eos_token_id {
            return Ok(GenerationEvent::GenerationStopped);
        }
        self.set_input_tokens(&[sampling_result.token]);

        if let Some(reasoning_supervisor) = self.reasoning_supervisor.as_mut() {
            match reasoning_supervisor.advance(&[sampling_result.token], &self.ongoing_gen_type) {
                GenerationEvent::ReasoningStarted => {
                    self.ongoing_gen_type = GenerationDataType::Reasoning;
                }
                GenerationEvent::ReasoningStopped => {
                    self.ongoing_gen_type = GenerationDataType::DirectContent;
                    return Ok(GenerationEvent::ReasoningSampled { sampling_result });
                }
                GenerationEvent::ForceTokens { tokens } => {
                    self.tokens_to_force.extend(tokens);
                }
                _ => {}
            }

            if self.ongoing_gen_type == GenerationDataType::Reasoning {
                return Ok(GenerationEvent::ReasoningSampled { sampling_result });
            }
        }

        if let Some(tool_calling_supervisor) = self.tool_calling_supervisor.as_mut() {
            match tool_calling_supervisor
                .advance(&[sampling_result.token], &self.ongoing_gen_type)?
            {
                GenerationEvent::ToolCallStarted => {
                    self.ongoing_gen_type = GenerationDataType::ToolCallName;
                    return Ok(GenerationEvent::ToolCallStarted);
                }
                GenerationEvent::ToolCallStopped { chunk } => {
                    let event = if let Some(chunk) = chunk {
                        if self.ongoing_gen_type == GenerationDataType::ToolCallName {
                            GenerationEvent::ToolCallNameChunk { chunk }
                        } else {
                            GenerationEvent::ToolCallArgumentsChunk { chunk }
                        }
                    } else {
                        GenerationEvent::ToolCallStopped { chunk: None }
                    };
                    self.ongoing_gen_type = GenerationDataType::DirectContent;
                    return Ok(event);
                }
                GenerationEvent::ToolCallNameChunk { chunk } => {
                    self.ongoing_gen_type = GenerationDataType::ToolCallArguments;
                    return Ok(GenerationEvent::ToolCallNameChunk { chunk });
                }
                GenerationEvent::ToolCallArgumentsChunk { chunk } => {
                    return Ok(GenerationEvent::ToolCallArgumentsChunk { chunk });
                }
                _ => {}
            }
        }

        self.force_tool_choice();

        if self.ongoing_gen_type == GenerationDataType::DirectContent {
            return Ok(GenerationEvent::ContentSampled { sampling_result });
        }

        Ok(GenerationEvent::None)
    }

    pub fn soft_stop(&mut self) {
        self.tokens_to_force.push(self.eos_token_id);
    }

    pub fn force_tool_choice(&mut self) {
        if self.ongoing_gen_type == GenerationDataType::DirectContent
            && let Some(template) = self.tool_choice_template.take()
        {
            self.tokens_to_force.extend(template);
        }
    }

    pub fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        if let Some(tool_calling_supervisor) = self.tool_calling_supervisor.as_ref() {
            tool_calling_supervisor.tool_calls()
        } else {
            None
        }
    }
}
