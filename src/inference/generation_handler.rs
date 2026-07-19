use crate::inference::{
    GenerationDataType, GenerationEvent, ReasoningSupervisorInterface, Sampler, SamplingResult,
    ToolCallingSupervisorInterface,
};
use crate::types::ToolCall;

use candle_core::Tensor;

pub struct GenerationHandler<'a> {
    pub input_tokens: Vec<u32>,
    pub sampler: Box<dyn Sampler>,
    pub reasoning_supervisor: Option<Box<dyn ReasoningSupervisorInterface>>,
    pub tool_calling_supervisor: Option<Box<dyn ToolCallingSupervisorInterface + 'a>>,
    pub ongoing_gen_type: GenerationDataType,
    pub tokens_to_force: Vec<u32>,
    pub eos_token_id: u32,
    pub tool_choice_template: Option<Vec<u32>>,
}

impl<'a> GenerationHandler<'a> {
    pub fn new(
        input_tokens: &[u32],
        sampler: Box<dyn Sampler>,
        mut reasoning_supervisor: Option<Box<dyn ReasoningSupervisorInterface>>,
        mut tool_calling_supervisor: Option<Box<dyn ToolCallingSupervisorInterface + 'a>>,
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

    pub fn reasoning_tokens_count(&self) -> u32 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::reasoning_supervisor::MockReasoningSupervisorInterface;
    use crate::inference::sampler::MockSampler;
    use crate::inference::tool_calling_supervisor::MockToolCallingSupervisorInterface;
    use crate::types::{ChunkFunctionCall, ChunkToolCall, FunctionDefinition};
    use mockall::predicate;
    use rstest::rstest;
    use std::collections::HashMap;

    fn name_chunk(id: &str, name: &str) -> ChunkToolCall {
        ChunkToolCall {
            index: 0,
            id: Some(id.to_string()),
            r#type: Some("function".to_string()),
            function: ChunkFunctionCall {
                name: Some(name.to_string()),
                arguments: String::new(),
            },
        }
    }

    fn args_chunk(args: &str) -> ChunkToolCall {
        ChunkToolCall {
            index: 0,
            id: None,
            r#type: None,
            function: ChunkFunctionCall {
                name: None,
                arguments: args.to_string(),
            },
        }
    }

    #[rstest]
    #[case(GenerationEvent::None, GenerationEvent::None, None, GenerationDataType::DirectContent, vec![])]
    #[case(GenerationEvent::None, GenerationEvent::None, Some(vec![21, 21]), GenerationDataType::DirectContent, vec![21, 21])]
    #[case(GenerationEvent::ReasoningStarted, GenerationEvent::None, None, GenerationDataType::Reasoning, vec![])]
    #[case(GenerationEvent::ReasoningStopped, GenerationEvent::None, None, GenerationDataType::DirectContent, vec![])]
    #[case(GenerationEvent::ForceTokens { tokens: vec![21, 21] }, GenerationEvent::None, None, GenerationDataType::DirectContent, vec![21, 21])]
    #[case(GenerationEvent::None, GenerationEvent::ToolCallStarted, None, GenerationDataType::ToolCallName, vec![])]
    #[case(GenerationEvent::ForceTokens { tokens: vec![21, 21] }, GenerationEvent::ToolCallStarted, None, GenerationDataType::ToolCallName, vec![21, 21])]
    #[case(GenerationEvent::None, GenerationEvent::ToolCallStarted, Some(vec![21, 21]), GenerationDataType::ToolCallName, vec![])]
    fn test_handler_init(
        #[case] reasoning_advance_output: GenerationEvent,
        #[case] tool_calling_advance_output: GenerationEvent,
        #[case] tool_choice_template: Option<Vec<u32>>,
        #[case] expected_ongoing_gen_type: GenerationDataType,
        #[case] expected_tokens_to_force: Vec<u32>,
    ) {
        let mut reasoning_supervisor = MockReasoningSupervisorInterface::new();
        reasoning_supervisor
            .expect_advance()
            .returning(move |_, _| reasoning_advance_output.clone());
        let mut tool_calling_supervisor = MockToolCallingSupervisorInterface::new();
        tool_calling_supervisor
            .expect_advance()
            .returning(move |_, _| Ok(tool_calling_advance_output.clone()));
        let handler = GenerationHandler::new(
            &vec![1, 2, 3],
            Box::new(MockSampler::new()),
            Some(Box::new(reasoning_supervisor)),
            Some(Box::new(tool_calling_supervisor)),
            4,
            tool_choice_template,
        );
        assert!(handler.is_ok());
        let handler = handler.unwrap();
        assert_eq!(handler.ongoing_gen_type, expected_ongoing_gen_type);
        assert_eq!(handler.tokens_to_force, expected_tokens_to_force);
    }

    #[rstest]
    // Parameters: (tokens_to_force, ongoing_gen_type, reasoning_output, tool_calling_output,
    //               sample_output, tool_choice_template, sample_times,
    //               expected_event, expected_gen_type, expected_force)
    #[case(
        vec![],
        GenerationDataType::DirectContent,
        GenerationEvent::None,
        GenerationEvent::None,
        SamplingResult { token: 55, logprob: None, top_k_logprobs: None },
        None,
        1,
        GenerationEvent::ContentSampled { sampling_result: SamplingResult { token: 55, logprob: None, top_k_logprobs: None } },
        GenerationDataType::DirectContent,
        vec![]
    )]
    #[case(
        vec![21, 21],
        GenerationDataType::DirectContent,
        GenerationEvent::None,
        GenerationEvent::None,
        SamplingResult { token: 55, logprob: None, top_k_logprobs: None },
        None,
        0,
        GenerationEvent::ContentSampled { sampling_result: SamplingResult { token: 21, logprob: None, top_k_logprobs: None } },
        GenerationDataType::DirectContent,
        vec![21]
    )]
    #[case(
        vec![],
        GenerationDataType::DirectContent,
        GenerationEvent::None,
        GenerationEvent::None,
        SamplingResult { token: 4, logprob: None, top_k_logprobs: None },
        None,
        1,
        GenerationEvent::GenerationStopped,
        GenerationDataType::DirectContent,
        vec![]
    )]
    #[case(
        vec![],
        GenerationDataType::DirectContent,
        GenerationEvent::ReasoningStarted,
        GenerationEvent::None,
        SamplingResult { token: 55, logprob: None, top_k_logprobs: None },
        None,
        1,
        GenerationEvent::ReasoningSampled { sampling_result: SamplingResult { token: 55, logprob: None, top_k_logprobs: None } },
        GenerationDataType::Reasoning,
        vec![]
    )]
    #[case(
        vec![],
        GenerationDataType::Reasoning,
        GenerationEvent::ReasoningStopped,
        GenerationEvent::None,
        SamplingResult { token: 55, logprob: None, top_k_logprobs: None },
        None,
        1,
        GenerationEvent::ReasoningSampled { sampling_result: SamplingResult { token: 55, logprob: None, top_k_logprobs: None } },
        GenerationDataType::DirectContent,
        vec![]
    )]
    #[case(
        vec![],
        GenerationDataType::Reasoning,
        GenerationEvent::ForceTokens { tokens: vec![21] },
        GenerationEvent::None,
        SamplingResult { token: 55, logprob: None, top_k_logprobs: None },
        None,
        1,
        GenerationEvent::ReasoningSampled { sampling_result: SamplingResult { token: 55, logprob: None, top_k_logprobs: None } },
        GenerationDataType::Reasoning,
        vec![21]
    )]
    #[case(
        vec![],
        GenerationDataType::DirectContent,
        GenerationEvent::None,
        GenerationEvent::ToolCallStarted,
        SamplingResult { token: 55, logprob: None, top_k_logprobs: None },
        None,
        1,
        GenerationEvent::ToolCallStarted,
        GenerationDataType::ToolCallName,
        vec![]
    )]
    #[case(
        vec![],
        GenerationDataType::ToolCallName,
        GenerationEvent::None,
        GenerationEvent::ToolCallStopped { chunk: None },
        SamplingResult { token: 55, logprob: None, top_k_logprobs: None },
        None,
        1,
        GenerationEvent::ToolCallStopped { chunk: None },
        GenerationDataType::DirectContent,
        vec![]
    )]
    #[case(
        vec![],
        GenerationDataType::ToolCallName,
        GenerationEvent::None,
        GenerationEvent::ToolCallStopped { chunk: Some(name_chunk("id", "name")) },
        SamplingResult { token: 55, logprob: None, top_k_logprobs: None },
        None,
        1,
        GenerationEvent::ToolCallNameChunk { chunk: name_chunk("id", "name") },
        GenerationDataType::DirectContent,
        vec![]
    )]
    #[case(
        vec![],
        GenerationDataType::ToolCallArguments,
        GenerationEvent::None,
        GenerationEvent::ToolCallStopped { chunk: Some(args_chunk("some arguments")) },
        SamplingResult { token: 55, logprob: None, top_k_logprobs: None },
        None,
        1,
        GenerationEvent::ToolCallArgumentsChunk { chunk: args_chunk("some arguments") },
        GenerationDataType::DirectContent,
        vec![]
    )]
    #[case(
        vec![],
        GenerationDataType::ToolCallName,
        GenerationEvent::None,
        GenerationEvent::ToolCallNameChunk { chunk: name_chunk("id", "name") },
        SamplingResult { token: 55, logprob: None, top_k_logprobs: None },
        None,
        1,
        GenerationEvent::ToolCallNameChunk { chunk: name_chunk("id", "name") },
        GenerationDataType::ToolCallArguments,
        vec![]
    )]
    #[case(
        vec![],
        GenerationDataType::ToolCallArguments,
        GenerationEvent::None,
        GenerationEvent::ToolCallArgumentsChunk { chunk: args_chunk("some arguments") },
        SamplingResult { token: 55, logprob: None, top_k_logprobs: None },
        None,
        1,
        GenerationEvent::ToolCallArgumentsChunk { chunk: args_chunk("some arguments") },
        GenerationDataType::ToolCallArguments,
        vec![]
    )]
    #[case(
        vec![],
        GenerationDataType::DirectContent,
        GenerationEvent::None,
        GenerationEvent::None,
        SamplingResult { token: 55, logprob: None, top_k_logprobs: None },
        Some(vec![1, 2, 3]),
        1,
        GenerationEvent::ContentSampled { sampling_result: SamplingResult { token: 55, logprob: None, top_k_logprobs: None } },
        GenerationDataType::DirectContent,
        vec![1, 2, 3]
    )]
    #[case(
        vec![],
        GenerationDataType::ToolCallName,
        GenerationEvent::None,
        GenerationEvent::None,
        SamplingResult { token: 55, logprob: None, top_k_logprobs: None },
        None,
        1,
        GenerationEvent::None,
        GenerationDataType::ToolCallName,
        vec![],
    )]
    fn test_advance(
        #[case] tokens_to_force: Vec<u32>,
        #[case] ongoing_gen_type: GenerationDataType,
        #[case] reasoning_advance_output: GenerationEvent,
        #[case] tool_calling_advance_output: GenerationEvent,
        #[case] sample_output: SamplingResult,
        #[case] tool_choice_template: Option<Vec<u32>>,
        #[case] expected_times_sample_called: usize,
        #[case] expected_event: GenerationEvent,
        #[case] expected_ongoing_gen_type: GenerationDataType,
        #[case] expected_tokens_to_force: Vec<u32>,
    ) {
        let mut reasoning_supervisor = MockReasoningSupervisorInterface::new();
        reasoning_supervisor
            .expect_advance()
            .returning(move |_, _| reasoning_advance_output.clone());
        let mut tool_calling_supervisor = MockToolCallingSupervisorInterface::new();
        tool_calling_supervisor
            .expect_advance()
            .returning(move |_, _| Ok(tool_calling_advance_output.clone()));
        let mut sampler = MockSampler::new();
        let sampler_output = sample_output.clone();
        sampler
            .expect_sample()
            .times(expected_times_sample_called)
            .returning(move |_, _| Ok(sampler_output.clone()));
        let mut handler = GenerationHandler {
            input_tokens: vec![],
            sampler: Box::new(sampler),
            ongoing_gen_type,
            tokens_to_force: tokens_to_force.clone(),
            reasoning_supervisor: Some(Box::new(reasoning_supervisor)),
            tool_calling_supervisor: Some(Box::new(tool_calling_supervisor)),
            tool_choice_template,
            eos_token_id: 4,
        };
        let event = handler.advance(
            &Tensor::zeros(3, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
            false,
            None,
        );
        assert!(event.is_ok());
        let event = event.unwrap();
        assert_eq!(event, expected_event);
        assert_eq!(handler.ongoing_gen_type, expected_ongoing_gen_type);
        assert_eq!(handler.tokens_to_force, expected_tokens_to_force);
        if event != GenerationEvent::GenerationStopped {
            let next_token = tokens_to_force.get(0).unwrap_or(&sample_output.token);
            assert_eq!(handler.input_tokens, vec![*next_token]);
        }
    }

    #[test]
    fn test_advance_with_no_supervisors() {
        let mut sampler = MockSampler::new();
        let sampling_result = SamplingResult {
            token: 55,
            logprob: None,
            top_k_logprobs: None,
        };
        sampler
            .expect_sample()
            .returning(move |_, _| Ok(sampling_result.clone()));
        let mut handler = GenerationHandler {
            input_tokens: vec![],
            sampler: Box::new(sampler),
            ongoing_gen_type: GenerationDataType::DirectContent,
            tokens_to_force: vec![],
            reasoning_supervisor: None,
            tool_calling_supervisor: None,
            tool_choice_template: None,
            eos_token_id: 4,
        };
        let event = handler.advance(
            &Tensor::zeros(3, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
            false,
            None,
        );
        assert!(event.is_ok());
        let event = event.unwrap();
        assert_eq!(
            event,
            GenerationEvent::ContentSampled {
                sampling_result: SamplingResult {
                    token: 55,
                    logprob: None,
                    top_k_logprobs: None
                }
            }
        );
        assert_eq!(handler.ongoing_gen_type, GenerationDataType::DirectContent);
    }

    #[rstest]
    #[case(false, None)]
    #[case(true, None)]
    #[case(true, Some(5))]
    fn test_advance_logprobs(#[case] with_logprobs: bool, #[case] top_k_logprobs: Option<usize>) {
        let mut sampler = MockSampler::new();
        let expected_top_k = if with_logprobs {
            Some(top_k_logprobs.unwrap_or(0))
        } else {
            None
        };
        sampler
            .expect_sample()
            .with(predicate::always(), predicate::eq(expected_top_k))
            .times(1)
            .returning(|_, _| {
                Ok(SamplingResult {
                    token: 55,
                    logprob: None,
                    top_k_logprobs: None,
                })
            });
        let mut handler = GenerationHandler {
            input_tokens: vec![],
            sampler: Box::new(sampler),
            ongoing_gen_type: GenerationDataType::DirectContent,
            tokens_to_force: vec![],
            reasoning_supervisor: None,
            tool_calling_supervisor: None,
            tool_choice_template: None,
            eos_token_id: 4,
        };
        let event = handler.advance(
            &Tensor::zeros(3, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
            with_logprobs,
            top_k_logprobs,
        );
        assert!(event.is_ok());
    }

    #[test]
    fn test_soft_stop() {
        let mut handler = GenerationHandler {
            input_tokens: vec![],
            sampler: Box::new(MockSampler::new()),
            ongoing_gen_type: GenerationDataType::DirectContent,
            tokens_to_force: vec![],
            reasoning_supervisor: None,
            tool_calling_supervisor: None,
            tool_choice_template: None,
            eos_token_id: 4,
        };
        handler.soft_stop();
        assert_eq!(handler.tokens_to_force, vec![4]);
    }

    #[rstest]
    #[case(GenerationDataType::DirectContent, None, vec![])]
    #[case(GenerationDataType::Reasoning, Some(vec![1, 2, 3]), vec![])]
    #[case(GenerationDataType::DirectContent, Some(vec![1, 2, 3]), vec![1, 2, 3])]
    fn test_force_tool_choice(
        #[case] ongoing_gen_type: GenerationDataType,
        #[case] tool_choice_template: Option<Vec<u32>>,
        #[case] expected_tokens_to_force: Vec<u32>,
    ) {
        let mut handler = GenerationHandler {
            input_tokens: vec![],
            sampler: Box::new(MockSampler::new()),
            ongoing_gen_type,
            tokens_to_force: vec![],
            reasoning_supervisor: None,
            tool_calling_supervisor: None,
            tool_choice_template,
            eos_token_id: 4,
        };
        handler.force_tool_choice();
        assert_eq!(handler.tokens_to_force, expected_tokens_to_force);
    }

    #[rstest]
    #[case(None, None)]
    #[case(Some({
        let mut mock = MockToolCallingSupervisorInterface::new();
        mock.expect_tool_calls()
            .times(1)
            .returning(move || {
                Some(vec![ToolCall {
                    id: String::from("id"),
                    r#type: String::from("function"),
                    function: FunctionDefinition {
                        name: String::from("name"),
                        arguments: HashMap::new(),
                    },
                }])
        });
        Box::new(mock) as Box<dyn ToolCallingSupervisorInterface>
    }),
    Some(vec![ToolCall {
        id: String::from("id"),
        r#type: String::from("function"),
        function: FunctionDefinition {
            name: String::from("name"),
            arguments: HashMap::new(),
        },
    }]))]
    fn test_tool_calls(
        #[case] tool_calling_supervisor: Option<Box<dyn ToolCallingSupervisorInterface>>,
        #[case] expected: Option<Vec<ToolCall>>,
    ) {
        let handler = GenerationHandler {
            input_tokens: vec![],
            sampler: Box::new(MockSampler::new()),
            ongoing_gen_type: GenerationDataType::DirectContent,
            tokens_to_force: vec![],
            reasoning_supervisor: None,
            tool_calling_supervisor,
            tool_choice_template: None,
            eos_token_id: 4,
        };
        assert_eq!(handler.tool_calls(), expected);
    }

    #[rstest]
    #[case(None, 0)]
    #[case(Some({
        let mut mock = MockReasoningSupervisorInterface::new();
        mock.expect_reasoning_tokens_count()
            .times(1)
            .returning(move || 4);
        Box::new(mock) as Box<dyn ReasoningSupervisorInterface>
    }),
    4)]
    fn test_reasoning_tokens_count(
        #[case] reasoning_supervisor: Option<Box<dyn ReasoningSupervisorInterface>>,
        #[case] expected: u32,
    ) {
        let handler = GenerationHandler {
            input_tokens: vec![],
            sampler: Box::new(MockSampler::new()),
            ongoing_gen_type: GenerationDataType::DirectContent,
            tokens_to_force: vec![],
            reasoning_supervisor: reasoning_supervisor,
            tool_calling_supervisor: None,
            tool_choice_template: None,
            eos_token_id: 4,
        };
        assert_eq!(handler.reasoning_tokens_count(), expected);
    }
}
