use crate::model::MixedCache;
use crate::inference::{InferenceSampler, ReasoningSupervisor, ToolCallingSupervisor, GenerationDataType, GenerationEvent, SamplingResult};
use crate::api::types::{ChunkToolCall, ToolCall};

use candle_core::Tensor;

pub struct GenerationHandler<'a> {
    pub input_tokens: Vec<u32>,
    pub sampler: InferenceSampler,
    pub reasoning_supervisor: Option<ReasoningSupervisor>,
    pub tool_calling_supervisor: Option<ToolCallingSupervisor<'a>>,
    pub ongoing_gen_type: GenerationDataType,
    pub tokens_to_force: Vec<u32>,
    pub eos_token_id: u32,
    pub pending_events: Vec<GenerationEvent>,
}

impl<'a> GenerationHandler<'a> {
    pub fn new(
        input_tokens: &[u32],
        mut sampler: InferenceSampler,
        mut reasoning_supervisor: Option<ReasoningSupervisor>,
        mut tool_calling_supervisor: Option<ToolCallingSupervisor<'a>>,
        eos_token_id: u32,
    )-> Self {
        let mut ongoing_gen_type = GenerationDataType::DirectContent;
        let mut tokens_to_force = vec![];
        if let Some(reasoning_supervisor) = reasoning_supervisor.as_mut() {
            reasoning_supervisor.advance(&input_tokens, &ongoing_gen_type);
            let events = reasoning_supervisor.take_events();
            for event in events {
                match event {
                    GenerationEvent::ReasoningStarted { .. } => {
                        ongoing_gen_type = GenerationDataType::Reasoning;
                    }
                    GenerationEvent::ReasoningStopped { .. } => {
                        ongoing_gen_type = GenerationDataType::DirectContent;
                    }
                    GenerationEvent::ForceTokens { tokens } => {
                        tokens_to_force.extend(tokens);
                    }
                    _ => {}
                }
            }
        }

        if let Some(tool_calling_supervisor) = tool_calling_supervisor.as_mut() {
            tool_calling_supervisor.advance(&input_tokens, &ongoing_gen_type);
            let events = tool_calling_supervisor.take_events();
            for event in events {
                match event {
                    GenerationEvent::ToolCallStarted => {
                        ongoing_gen_type = GenerationDataType::ToolCallName;
                    }
                    GenerationEvent::ForceTokens { tokens } => {
                        tokens_to_force.extend(tokens);
                    }
                    _ => {}
                }
            }
        }


        Self {
            input_tokens: input_tokens.to_vec(),
            sampler,
            reasoning_supervisor,
            tool_calling_supervisor,
            ongoing_gen_type,
            tokens_to_force,
            eos_token_id,
            pending_events: Vec::with_capacity(4),
        }
    }

    pub fn gen_type(&self) -> GenerationDataType {
        self.ongoing_gen_type.clone()
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

    pub fn advance(&mut self, logits: &Tensor, with_logprobs: bool, top_k_logprobs: Option<usize>) -> anyhow::Result<()> {
        let sampling_result = if self.tokens_to_force.is_empty() {
            if with_logprobs {
                self.sampler.sample_with_logprobs(logits, top_k_logprobs.unwrap_or(0))?
            } else {
                SamplingResult {
                    token: self.sampler.sample(logits)?,
                    logprob: None,
                    top_k_logprobs: None,
                }
            }
        } else {
            SamplingResult {
                token: self.tokens_to_force.remove(0),
                logprob: None,
                top_k_logprobs: None,
            }
        };

        if sampling_result.token == self.eos_token_id {
            self.emit_event(GenerationEvent::GenerationStopped);
            return Ok(()); 
        }
        self.set_input_tokens(&[sampling_result.token]);

        if let Some(reasoning_supervisor) = self.reasoning_supervisor.as_mut() {
            reasoning_supervisor.advance(&[sampling_result.token], &self.ongoing_gen_type);
        }
        self.handle_reasoning_events()?;

        if let Some(tool_calling_supervisor) = self.tool_calling_supervisor.as_mut() {
            tool_calling_supervisor.advance(&[sampling_result.token], &self.ongoing_gen_type);
        }

        self.handle_tool_call_events()?;

        if self.pending_events.iter().any(|event| matches!(event, GenerationEvent::ReasoningStopped | GenerationEvent::ToolCallStopped)) {
            return Ok(()); 
        }

        match self.ongoing_gen_type {
            GenerationDataType::DirectContent => {
                self.emit_event(GenerationEvent::ContentSampled { sampling_result });
            }
            GenerationDataType::Reasoning => {
                self.emit_event(GenerationEvent::ReasoningSampled { sampling_result });
            }
            _ => {}
        }

        Ok(())

    }

    pub fn soft_stop(&mut self) {
        self.tokens_to_force.push(self.eos_token_id);
    }

    pub fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        if let Some(tool_calling_supervisor) = self.tool_calling_supervisor.as_ref() {
            tool_calling_supervisor.tool_calls()
        } else {
            None
        }
    }

    fn transition_to_tool_arguments(&mut self) {
        self.ongoing_gen_type = GenerationDataType::ToolCallArguments;
    }

    fn handle_reasoning_events(&mut self) -> anyhow::Result<()> {
        if let Some(reasoning_supervisor) = self.reasoning_supervisor.as_mut() {
            let events = reasoning_supervisor.take_events();
            for event in events {
                match event {
                    GenerationEvent::ReasoningStarted { .. } => {
                        self.ongoing_gen_type = GenerationDataType::Reasoning;
                    }
                    GenerationEvent::ReasoningStopped { .. } => {
                        self.ongoing_gen_type = GenerationDataType::DirectContent;
                        self.pending_events.push(event);
                    }
                    GenerationEvent::ForceTokens { tokens } => {
                        self.tokens_to_force.extend(tokens);
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Collect pending events from supervisors and process them
    fn handle_tool_call_events(&mut self) -> anyhow::Result<()> {
        if let Some(tool_calling_supervisor) = self.tool_calling_supervisor.as_mut() {
            let events = tool_calling_supervisor.take_events();
            for event in events {
                match event {
                    GenerationEvent::ToolCallStarted => {
                        self.ongoing_gen_type = GenerationDataType::ToolCallName;
                    }
                    GenerationEvent::ToolCallStopped => {
                        self.ongoing_gen_type = GenerationDataType::DirectContent;
                    }
                    GenerationEvent::ToolCallNameChunk { .. } => {
                        self.ongoing_gen_type = GenerationDataType::ToolCallArguments
                    }
                    _ => {}
                }
                self.pending_events.push(event);
            }
        }
        Ok(())
    }

    /// Get all pending supervisor events to be sent to callback
    pub fn take_pending_events(&mut self) -> Vec<GenerationEvent> {
        std::mem::take(&mut self.pending_events)
    }

    fn emit_event(&mut self, event: GenerationEvent) {
        self.pending_events.push(event);
    }
}