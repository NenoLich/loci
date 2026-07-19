use crate::inference::{
    GenerationDataType, GenerationEvent, PostSamplingConfig, ToolArgFormatter,
    ToolArgFormatterBuilder,
};
use crate::tokenizer::{StreamContext, Tokenizer};
use crate::types::{
    ChunkFunctionCall, ChunkToolCall, FunctionDefinition, SpecificToolChoice, ToolCall, ToolChoice,
    ToolChoiceMode,
};
#[cfg(test)]
use mockall::automock;
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, PartialEq, Clone)]
pub enum StateEnum {
    None,
    NamePrefix,
    NameContent,
    ArgumentsPrefix,
    ArgumentsContent,
}

impl StateEnum {
    fn next_state(&self, context: &mut ToolCallingSupervisorContext) -> anyhow::Result<StateEnum> {
        match self {
            StateEnum::None => NoneState.next_state(context),
            StateEnum::NamePrefix => NamePrefixState.next_state(context),
            StateEnum::NameContent => NameContentState.next_state(context),
            StateEnum::ArgumentsPrefix => ArgumentsPrefixState.next_state(context),
            StateEnum::ArgumentsContent => ArgumentsContentState.next_state(context),
        }
    }
}

pub trait ToolCallParsingState {
    fn next_state(&self, context: &mut ToolCallingSupervisorContext) -> anyhow::Result<StateEnum>;
}

pub struct NoneState;
impl ToolCallParsingState for NoneState {
    fn next_state(&self, context: &mut ToolCallingSupervisorContext) -> anyhow::Result<StateEnum> {
        NamePrefixState.next_state(context)
    }
}

pub struct NamePrefixState;
impl ToolCallParsingState for NamePrefixState {
    fn next_state(&self, context: &mut ToolCallingSupervisorContext) -> anyhow::Result<StateEnum> {
        let decoded_tokens_option = context
            .tokenizer
            .process_multiple_token_stream(&mut context.stream_ctx, &context.current_ids_buffer)?;
        context.current_ids_buffer.clear();
        if let Some(decoded_tokens) = decoded_tokens_option {
            context.decoded_scratchpad.push_str(&decoded_tokens);

            if context
                .formatter
                .try_strip_name_prefix(&mut context.decoded_scratchpad)
            {
                return NameContentState.next_state(context);
            }
        }

        Ok(StateEnum::NamePrefix)
    }
}

pub struct NameContentState;
impl ToolCallParsingState for NameContentState {
    fn next_state(&self, context: &mut ToolCallingSupervisorContext) -> anyhow::Result<StateEnum> {
        let tool_name_option = context.formatter.try_extract_function_name(
            &context.current_ids_buffer,
            &mut context.decoded_scratchpad,
            context.tokenizer,
            &mut context.stream_ctx,
        )?;
        context.current_ids_buffer.clear();
        match tool_name_option {
            Some(tool_name) => {
                context.set_tool_name(tool_name);
                context.current_tool_id = Uuid::new_v4().to_string();

                ArgumentsPrefixState.next_state(context)
            }
            None => Ok(StateEnum::NameContent),
        }
    }
}

pub struct ArgumentsPrefixState;
impl ToolCallParsingState for ArgumentsPrefixState {
    fn next_state(&self, context: &mut ToolCallingSupervisorContext) -> anyhow::Result<StateEnum> {
        let decoded_tokens_option = context
            .tokenizer
            .process_multiple_token_stream(&mut context.stream_ctx, &context.current_ids_buffer)?;
        context.current_ids_buffer.clear();
        if let Some(decoded_tokens) = decoded_tokens_option {
            context.decoded_scratchpad.push_str(&decoded_tokens);

            if context
                .formatter
                .try_strip_arguments_prefix(&mut context.decoded_scratchpad)
            {
                return ArgumentsContentState.next_state(context);
            }
        }

        Ok(StateEnum::ArgumentsPrefix)
    }
}

pub struct ArgumentsContentState;
impl ToolCallParsingState for ArgumentsContentState {
    fn next_state(&self, context: &mut ToolCallingSupervisorContext) -> anyhow::Result<StateEnum> {
        // Stream decode arguments using the efficient circular buffer
        context.format_and_decode_args()?;
        context.fix_and_update_args()?;

        Ok(StateEnum::ArgumentsContent)
    }
}

pub struct ToolCallingSupervisorContext<'a> {
    pub formatter: Box<dyn ToolArgFormatter>,
    pub tokenizer: &'a dyn Tokenizer,
    pub stream_ctx: StreamContext, // Streaming context for tool arguments
    pub tool_call_start_token_id: u32,
    pub tool_call_end_token_id: u32,
    pub tool_calls: Vec<ToolCall>,
    pub current_tool_id: String,
    pub current_tool_name: String,
    pub current_tool_arguments: String,
    pub current_ids_buffer: Vec<u32>,
    pub decoded_scratchpad: String,
    pub stream_args_pos: usize,
}

impl<'a> ToolCallingSupervisorContext<'a> {
    fn set_tool_name(&mut self, name: String) {
        self.current_tool_name = name;
        self.current_tool_name
            .retain(|c| c.is_alphanumeric() || c == '_');
    }
    fn format_and_decode_args(&mut self) -> anyhow::Result<()> {
        self.formatter.format_args(
            &self.current_ids_buffer,
            &mut self.decoded_scratchpad,
            self.tokenizer,
            &mut self.stream_ctx,
        )?;
        self.current_ids_buffer.clear();
        Ok(())
    }

    fn fix_and_update_args(&mut self) -> anyhow::Result<()> {
        self.formatter.fix_json(
            &mut self.decoded_scratchpad,
            self.current_tool_arguments.is_empty(),
        );
        self.current_tool_arguments
            .push_str(&self.decoded_scratchpad);
        self.decoded_scratchpad.clear();
        Ok(())
    }
}

#[cfg_attr(test, automock)]
pub trait ToolCallingSupervisorInterface {
    fn advance(
        &mut self,
        input_tokens: &[u32],
        ongoing_gen_type: &GenerationDataType,
    ) -> anyhow::Result<GenerationEvent>;
    fn tool_calls(&self) -> Option<Vec<ToolCall>>;
}

pub struct ToolCallingSupervisor<'a> {
    pub context: ToolCallingSupervisorContext<'a>,
    pub tool_call_parsing_state: StateEnum,
}

impl<'a> ToolCallingSupervisorInterface for ToolCallingSupervisor<'a> {
    fn advance(
        &mut self,
        token_ids: &[u32],
        ongoing_gen_type: &GenerationDataType,
    ) -> anyhow::Result<GenerationEvent> {
        Ok(match ongoing_gen_type {
            GenerationDataType::DirectContent => {
                if self.detect_tool_call_start(token_ids) {
                    GenerationEvent::ToolCallStarted
                } else {
                    GenerationEvent::None
                }
            }
            GenerationDataType::ToolCallName | GenerationDataType::ToolCallArguments => {
                self.context.current_ids_buffer.extend_from_slice(token_ids);
                if self.detect_tool_call_end() {
                    let chunk = self.finalize_tool_call(ongoing_gen_type)?;
                    self.reset();
                    GenerationEvent::ToolCallStopped { chunk }
                } else {
                    self.tool_call_parsing_state =
                        self.tool_call_parsing_state.next_state(&mut self.context)?;
                    if let Some(event) = self.emit_tool_call_chunk_event(ongoing_gen_type) {
                        event
                    } else {
                        GenerationEvent::None
                    }
                }
            }
            _ => GenerationEvent::None,
        })
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        Some(self.context.tool_calls.clone())
    }
}

impl<'a> ToolCallingSupervisor<'a> {
    pub fn new(
        supports_tool_calling: bool,
        config: &PostSamplingConfig,
        tokenizer: &'a dyn Tokenizer,
    ) -> anyhow::Result<Option<Self>> {
        if !supports_tool_calling {
            return Ok(None);
        }

        let formatter = ToolArgFormatterBuilder::new(config)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build tool argument formatter: {}", e))?;

        let Some(tool_call_start_token_id) = config.tool_call_start_token_id else {
            return Ok(None);
        };
        let Some(tool_call_end_token_id) = config.tool_call_end_token_id else {
            return Ok(None);
        };

        let stream_ctx = StreamContext::with_capacity(16);

        let context = ToolCallingSupervisorContext {
            formatter,
            tokenizer,
            stream_ctx,
            tool_call_start_token_id,
            tool_call_end_token_id,
            tool_calls: Vec::new(),
            current_tool_id: String::new(),
            current_tool_name: String::new(),
            current_tool_arguments: String::new(),
            current_ids_buffer: Vec::with_capacity(50),
            decoded_scratchpad: String::with_capacity(50),
            stream_args_pos: 0,
        };

        Ok(Some(ToolCallingSupervisor {
            context,
            tool_call_parsing_state: StateEnum::None,
        }))
    }

    pub fn get_tool_choice_template(
        &self,
        tokenizer: &'a dyn Tokenizer,
        tool_choice: &ToolChoice,
    ) -> anyhow::Result<Option<Vec<u32>>> {
        match tool_choice {
            ToolChoice::Mode(ToolChoiceMode::Required) => {
                Ok(Some(self.context.formatter.build_tool_call_template(
                    &self.context.tool_call_start_token_id,
                    None,
                    tokenizer,
                )?))
            }
            ToolChoice::Specific(SpecificToolChoice { function, .. }) => {
                Ok(Some(self.context.formatter.build_tool_call_template(
                    &self.context.tool_call_start_token_id,
                    Some(&function.name),
                    tokenizer,
                )?))
            }
            _ => Ok(None),
        }
    }

    pub fn detect_tool_call_start(&mut self, token_ids: &[u32]) -> bool {
        token_ids.ends_with(std::slice::from_ref(&self.context.tool_call_start_token_id))
    }

    pub fn detect_tool_call_end(&self) -> bool {
        self.context
            .current_ids_buffer
            .ends_with(std::slice::from_ref(&self.context.tool_call_end_token_id))
    }

    pub fn finalize_tool_call(
        &mut self,
        ongoing_gen_type: &GenerationDataType,
    ) -> anyhow::Result<Option<ChunkToolCall>> {
        let chunk = match ongoing_gen_type {
            GenerationDataType::ToolCallName => {
                if self.context.current_tool_name.is_empty() {
                    // Corrected: Explicitly cast end sequence vector to a slice view reference
                    let suffix_slice = std::slice::from_ref(&self.context.tool_call_end_token_id);
                    let clean_ids = self
                        .context
                        .current_ids_buffer
                        .strip_suffix(suffix_slice)
                        .unwrap_or(&self.context.current_ids_buffer);

                    if !clean_ids.is_empty()
                        && let Some(tool_name) =
                            self.context.tokenizer.process_multiple_token_stream(
                                &mut self.context.stream_ctx,
                                clean_ids,
                            )?
                    {
                        self.context.decoded_scratchpad.push_str(&tool_name);
                    }
                    let name = std::mem::take(&mut self.context.decoded_scratchpad);
                    self.context.set_tool_name(name);
                }
                if self.context.current_tool_id.is_empty() {
                    self.context.current_tool_id = Uuid::new_v4().to_string();
                }
                self.context.current_ids_buffer.clear();
                Some(self.emit_initial_tool_call_chunk())
            }

            GenerationDataType::ToolCallArguments => {
                let suffix_slice = std::slice::from_ref(&self.context.tool_call_end_token_id);

                // Clean out the ending structural token IDs before processing the final token strings
                if self.context.current_ids_buffer.ends_with(suffix_slice) {
                    let new_len = self.context.current_ids_buffer.len() - suffix_slice.len();
                    self.context.current_ids_buffer.truncate(new_len);
                }

                // Decode the remaining pure argument tokens left in the buffer scratchpad
                if !self.context.current_ids_buffer.is_empty() {
                    self.context.format_and_decode_args()?;
                    self.context.fix_and_update_args()?;
                }

                // CRITICAL ARCHITECTURAL SAFETY NET:
                // Ensure the generated string is a complete, syntactically legal JSON payload.
                // If the model was interrupted or the custom formatter missed a trailing brace,
                // we patch the string safely before handing it to the serde parser.
                let current_args = self.context.current_tool_arguments.trim();
                if !current_args.is_empty() && !current_args.ends_with('}') {
                    self.context.current_tool_arguments.push('}');
                }
                self.emit_arg_tool_call_chunk()
            }
            _ => None,
        };

        let json_arguments: HashMap<String, Value> =
            if self.context.current_tool_arguments.is_empty() {
                HashMap::new()
            } else {
                serde_json::from_str(&self.context.current_tool_arguments).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to parse model tool arguments into JSON. Raw String: '{}'. Error: {}",
                    self.context.current_tool_arguments,
                    e
                )
            })?
            };

        let tool_call = ToolCall {
            id: self.context.current_tool_id.clone(),
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: self.context.current_tool_name.clone(),
                arguments: json_arguments,
            },
        };

        self.context.tool_calls.push(tool_call.clone());

        Ok(chunk)
    }

    pub fn reset(&mut self) {
        self.context.current_ids_buffer.clear();
        self.context.current_tool_name.clear();
        self.context.current_tool_id.clear();
        self.context.current_tool_arguments.clear();
        self.context.stream_args_pos = 0;
        self.context.formatter.reset();
        self.tool_call_parsing_state = StateEnum::None;
        // Reset the streaming context for next tool call
        self.context.stream_ctx.reset();
    }

    pub fn emit_tool_call_chunk_event(
        &mut self,
        ongoing_gen_type: &GenerationDataType,
    ) -> Option<GenerationEvent> {
        match ongoing_gen_type {
            GenerationDataType::ToolCallName => {
                if !self.context.current_tool_name.is_empty() {
                    Some(GenerationEvent::ToolCallNameChunk {
                        chunk: self.emit_initial_tool_call_chunk(),
                    })
                } else {
                    None
                }
            }
            GenerationDataType::ToolCallArguments => self
                .emit_arg_tool_call_chunk()
                .map(|chunk| GenerationEvent::ToolCallArgumentsChunk { chunk }),
            _ => None,
        }
    }

    pub fn emit_initial_tool_call_chunk(&mut self) -> ChunkToolCall {
        let id = Some(if self.context.current_tool_id.is_empty() {
            let new_id = Uuid::new_v4();
            self.context.current_tool_id = new_id.to_string();
            new_id.to_string()
        } else {
            self.context.current_tool_id.clone()
        });

        ChunkToolCall {
            index: 0,
            id,
            r#type: Some("function".to_string()),
            function: ChunkFunctionCall {
                name: Some(self.context.current_tool_name.clone()),
                arguments: String::new(),
            },
        }
    }

    pub fn emit_arg_tool_call_chunk(&mut self) -> Option<ChunkToolCall> {
        if self.context.current_tool_arguments.is_empty()
            || self.context.stream_args_pos >= self.context.current_tool_arguments.len()
        {
            return None;
        }
        let arguments =
            self.context.current_tool_arguments[self.context.stream_args_pos..].to_string();
        self.context.stream_args_pos = self.context.current_tool_arguments.len();
        Some(ChunkToolCall {
            index: 0,
            id: None,
            r#type: None,
            function: ChunkFunctionCall {
                name: None,
                arguments,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::tool_formatter::MockToolArgFormatter;
    use crate::tokenizer::MockTokenizer;
    use proptest::prelude::*;
    use rstest::rstest;

    fn make_context<'t>(
        tokenizer: &'t MockTokenizer,
        formatter: Box<dyn ToolArgFormatter>,
    ) -> ToolCallingSupervisorContext<'t> {
        let stream_ctx = StreamContext::with_capacity(16);
        let tool_call_start_token_id = 1;
        let tool_call_end_token_id = 2;
        ToolCallingSupervisorContext {
            formatter,
            tokenizer,
            stream_ctx,
            tool_call_start_token_id,
            tool_call_end_token_id,
            tool_calls: Vec::new(),
            current_tool_id: String::new(),
            current_tool_name: String::new(),
            current_tool_arguments: String::new(),
            current_ids_buffer: Vec::with_capacity(50),
            decoded_scratchpad: String::with_capacity(50),
            stream_args_pos: 0,
        }
    }

    fn setup_mock_formatter(
        try_strip_name_prefix_result: bool,
        try_extract_function_name_result: Option<String>,
        try_strip_arguments_prefix_result: bool,
    ) -> Box<dyn ToolArgFormatter> {
        let mut formatter = MockToolArgFormatter::new();
        formatter
            .expect_try_strip_name_prefix()
            .returning(move |_| try_strip_name_prefix_result);
        formatter
            .expect_try_extract_function_name()
            .returning(move |_, _, _, _| Ok(try_extract_function_name_result.clone()));
        formatter
            .expect_try_strip_arguments_prefix()
            .returning(move |_| try_strip_arguments_prefix_result);
        formatter
            .expect_format_args()
            .returning(move |_, _, _, _| Ok(()));
        formatter.expect_fix_json().returning(move |_, _| ());
        formatter.expect_reset().returning(move || ());

        Box::new(formatter)
    }

    #[rstest]
    #[case(
        StateEnum::None,
        Some(String::from("output")),
        false,
        None,
        false,
        StateEnum::NamePrefix
    )]
    #[case(
        StateEnum::NamePrefix,
        Some(String::from("output")),
        true,
        None,
        false,
        StateEnum::NameContent
    )]
    #[case(StateEnum::NamePrefix, Some(String::from("output")), true, Some("some_tool".to_string()), false, StateEnum::ArgumentsPrefix)]
    #[case(
        StateEnum::NamePrefix,
        Some(String::from("output")),
        false,
        None,
        false,
        StateEnum::NamePrefix
    )]
    #[case(StateEnum::NamePrefix, None, true, None, false, StateEnum::NamePrefix)]
    #[case(StateEnum::NameContent, Some(String::from("output")), false, Some("some_tool".to_string()), false, StateEnum::ArgumentsPrefix)]
    #[case(
        StateEnum::NameContent,
        Some(String::from("output")),
        false,
        None,
        false,
        StateEnum::NameContent
    )]
    #[case(StateEnum::NameContent, None, false, Some("some_tool".to_string()), false, StateEnum::ArgumentsPrefix)]
    #[case(
        StateEnum::NameContent,
        None,
        false,
        None,
        false,
        StateEnum::NameContent
    )]
    #[case(
        StateEnum::ArgumentsPrefix,
        Some(String::from("output")),
        false,
        None,
        true,
        StateEnum::ArgumentsContent
    )]
    #[case(
        StateEnum::ArgumentsPrefix,
        Some(String::from("output")),
        false,
        None,
        false,
        StateEnum::ArgumentsPrefix
    )]
    #[case(
        StateEnum::ArgumentsPrefix,
        None,
        false,
        None,
        true,
        StateEnum::ArgumentsPrefix
    )]
    #[case(
        StateEnum::ArgumentsContent,
        Some(String::from("output")),
        false,
        None,
        true,
        StateEnum::ArgumentsContent
    )]
    #[case(
        StateEnum::ArgumentsContent,
        None,
        true,
        None,
        true,
        StateEnum::ArgumentsContent
    )]
    fn test_state_machine_transition(
        #[case] state: StateEnum,
        #[case] tokenizer_output: Option<String>,
        #[case] try_strip_name_prefix_result: bool,
        #[case] try_extract_function_name_result: Option<String>,
        #[case] try_strip_arguments_prefix_result: bool,
        #[case] next_state: StateEnum,
    ) {
        let mut tokenizer = MockTokenizer::new();
        tokenizer
            .expect_process_multiple_token_stream()
            .returning(move |_, _| Ok(tokenizer_output.clone()));
        let formatter = setup_mock_formatter(
            try_strip_name_prefix_result,
            try_extract_function_name_result,
            try_strip_arguments_prefix_result,
        );

        let mut context = make_context(&tokenizer, formatter);
        let result = state.next_state(&mut context);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result, next_state);
    }

    #[rstest]
    #[case(1, vec![3, 2, 1], true)]
    #[case(1, vec![1, 2, 3], false)]
    #[case(1, vec![2, 3], false)]
    #[case(1, vec![1], true)]
    #[case(1, vec![], false)]
    fn test_detect_tool_call_start(
        #[case] tool_call_start_token_id: u32,
        #[case] token_ids: Vec<u32>,
        #[case] expected_result: bool,
    ) {
        let tokenizer = MockTokenizer::new();
        let mut context = make_context(&tokenizer, Box::new(MockToolArgFormatter::new()));
        context.tool_call_start_token_id = tool_call_start_token_id;
        let mut supervisor = ToolCallingSupervisor {
            context: context,
            tool_call_parsing_state: StateEnum::None,
        };
        let result = supervisor.detect_tool_call_start(&token_ids);
        assert_eq!(result, expected_result);
    }

    #[rstest]
    #[case(2, vec![1, 2], true)]
    #[case(2, vec![1, 2, 3], false)]
    #[case(2, vec![1, 3], false)]
    #[case(2, vec![2], true)]
    #[case(2, vec![], false)]
    fn test_detect_tool_call_end(
        #[case] tool_call_end_token_id: u32,
        #[case] token_ids: Vec<u32>,

        #[case] expected_result: bool,
    ) {
        let tokenizer = MockTokenizer::new();
        let mut context = make_context(&tokenizer, Box::new(MockToolArgFormatter::new()));
        context.tool_call_start_token_id = tool_call_end_token_id;
        context.current_ids_buffer = token_ids;
        let supervisor = ToolCallingSupervisor {
            context: context,
            tool_call_parsing_state: StateEnum::None,
        };
        let result = supervisor.detect_tool_call_end();
        assert_eq!(result, expected_result);
    }

    #[rstest]
    // Parameters: (start_id, end_id, state, token_ids, strip_name, extract_name, strip_args,
    //               tokenizer_out, gen_type, expected_event, expected_reset)
    #[case(
        1,
        2,
        StateEnum::None,
        vec![],
        true,
        None,
        true,
        None,
        GenerationDataType::DirectContent,
        GenerationEvent::None,
        false,
    )]
    #[case(
        1,
        2,
        StateEnum::None,
        vec![3, 2],
        false,
        None,
        false,
        None,
        GenerationDataType::DirectContent,
        GenerationEvent::None,
        false,
    )]
    #[case(
        1,
        2,
        StateEnum::None,
        vec![3, 2, 1],
        false,
        None,
        false,
        None,
        GenerationDataType::DirectContent,
        GenerationEvent::ToolCallStarted,
        false,
    )]
    #[case(
        1,
        2,
        StateEnum::NameContent,
        vec![3, 222, 111],
        false,
        Some(String::from("foo")),
        false,
        Some(String::from("foo")),
        GenerationDataType::ToolCallName,
        GenerationEvent::ToolCallNameChunk { chunk: ChunkToolCall::default() },
        false,
    )]
    #[case(
        1,
        2,
        StateEnum::None,
        vec![3, 222, 111],
        true,
        Some(String::from("foo")),
        false,
        Some(String::from("foo")),
        GenerationDataType::ToolCallName,
        GenerationEvent::ToolCallNameChunk { chunk: ChunkToolCall::default() },
        false,
    )]
    #[case(
        1,
        2,
        StateEnum::ArgumentsContent,
        vec![3, 222, 111],
        false,
        Some(String::from("foo")),
        false,
        Some(String::from("foo")),
        GenerationDataType::ToolCallArguments,
        GenerationEvent::None,
        false,
    )]
    #[case(
        1,
        2,
        StateEnum::ArgumentsPrefix,
        vec![3, 222, 111],
        false,
        Some(String::from("foo")),
        true,
        Some(String::from("foo")),
        GenerationDataType::ToolCallArguments,
        GenerationEvent::ToolCallArgumentsChunk { chunk: ChunkToolCall::default() },
        false,
    )]
    #[case(
        1,
        2,
        StateEnum::ArgumentsPrefix,
        vec![3, 222, 111],
        false,
        Some(String::from("foo")),
        false,
        Some(String::from("foo")),
        GenerationDataType::ToolCallArguments,
        GenerationEvent::None,
        false,
    )]
    #[case(
        1,
        2,
        StateEnum::NamePrefix,
        vec![3, 222, 2],
        true,
        Some(String::from("foo")),
        false,
        Some(String::from("foo")),
        GenerationDataType::ToolCallName,
        GenerationEvent::ToolCallStopped { chunk: Some(ChunkToolCall::default()) },
        true,
    )]
    #[case(
        1,
        2,
        StateEnum::ArgumentsPrefix,
        vec![3, 222, 2],
        true,
        Some(String::from("foo")),
        true,
        Some(String::from("foo")),
        GenerationDataType::ToolCallArguments,
        GenerationEvent::ToolCallStopped { chunk: Some(ChunkToolCall::default()) },
        true,
    )]
    #[case(
        1,
        2,
        StateEnum::ArgumentsPrefix,
        vec![3, 222, 2],
        true,
        Some(String::from("foo")),
        true,
        None,
        GenerationDataType::ToolCallArguments,
        GenerationEvent::ToolCallStopped { chunk: None },
        true,
    )]
    fn test_advance(
        #[case] tool_call_start_token_id: u32,
        #[case] tool_call_end_token_id: u32,
        #[case] tool_call_parsing_state: StateEnum,
        #[case] token_ids: Vec<u32>,
        #[case] try_strip_name_prefix_result: bool,
        #[case] try_extract_function_name_result: Option<String>,
        #[case] try_strip_arguments_prefix_result: bool,
        #[case] tokenizer_output: Option<String>,
        #[case] ongoing_gen_type: GenerationDataType,
        #[case] expected_event: GenerationEvent,
        #[case] expected_reset: bool,
    ) {
        let mut tokenizer = MockTokenizer::new();
        tokenizer
            .expect_process_multiple_token_stream()
            .returning(move |_, _| Ok(tokenizer_output.clone()));
        let formatter = setup_mock_formatter(
            try_strip_name_prefix_result,
            try_extract_function_name_result,
            try_strip_arguments_prefix_result,
        );
        let mut context = make_context(&tokenizer, formatter);
        context.tool_call_start_token_id = tool_call_start_token_id;
        context.tool_call_end_token_id = tool_call_end_token_id;
        let mut supervisor = ToolCallingSupervisor {
            context: context,
            tool_call_parsing_state: tool_call_parsing_state,
        };
        let result = supervisor.advance(&token_ids, &ongoing_gen_type);
        assert!(result.is_ok());
        let result = result.unwrap();
        if std::mem::discriminant(&result) != std::mem::discriminant(&expected_event) {
            panic!("expected {:?}, got {:?}", expected_event, result);
        }
        match result {
            GenerationEvent::ToolCallNameChunk { chunk } => {
                assert!(!expected_reset, "reset should not have been called");
                assert!(!supervisor.context.current_tool_name.is_empty());
                assert!(!supervisor.context.current_tool_id.is_empty());
                assert_eq!(chunk.index, 0);
                assert_eq!(chunk.r#type, Some("function".to_string()));
                assert_eq!(chunk.function.arguments, String::new());
                assert_eq!(chunk.id, Some(supervisor.context.current_tool_id));
                assert_eq!(
                    chunk.function.name,
                    Some(supervisor.context.current_tool_name)
                );
            }
            GenerationEvent::ToolCallArgumentsChunk { chunk } => {
                assert!(!expected_reset, "reset should not have been called");
                assert!(!supervisor.context.current_tool_arguments.is_empty());
                assert_eq!(chunk.index, 0);
                assert!(chunk.r#type.is_none());
                assert!(chunk.function.name.is_none());
                assert!(!chunk.function.arguments.is_empty());
            }
            GenerationEvent::ToolCallStopped { chunk } => {
                assert!(expected_reset, "reset should have been called");
                assert!(supervisor.context.current_tool_name.is_empty());
                assert!(supervisor.context.current_tool_id.is_empty());
                match ongoing_gen_type {
                    GenerationDataType::ToolCallName => {
                        let chunk = chunk.unwrap();
                        assert_eq!(chunk.index, 0);
                        assert_eq!(chunk.r#type, Some("function".to_string()));
                        assert_eq!(chunk.function.arguments, String::new());
                        let last_tool_call = supervisor.context.tool_calls.last().unwrap();
                        assert_eq!(chunk.id, Some(last_tool_call.id.clone()));
                        assert_eq!(
                            chunk.function.name,
                            Some(last_tool_call.function.name.clone())
                        );
                    }
                    GenerationDataType::ToolCallArguments => {
                        assert!(supervisor.context.current_tool_arguments.is_empty());
                        if let Some(chunk) = chunk {
                            assert_eq!(chunk.index, 0);
                            assert!(chunk.r#type.is_none());
                            assert!(chunk.function.name.is_none());
                            assert!(!chunk.function.arguments.is_empty());
                        }
                    }
                    _ => (),
                }
            }
            _ => (),
        }
    }

    #[rstest]
    #[case(
        GenerationDataType::ToolCallName,
        vec![1, 2, 3],
        None,
        None,
        None,
        Some(ChunkToolCall::default()),
        Some(ToolCall {
            id: String::from("foo"),
            r#type: String::from("function"),
            function: FunctionDefinition {
                name: String::from("foo"),
                arguments: HashMap::new(),
            },
        }),
        None,
    )]
    #[case(
        GenerationDataType::DirectContent,
        vec![1, 2, 3],
        None,
        None,
        None,
        None,
        None,
        None,
    )]
    #[case(
        GenerationDataType::ToolCallArguments,
        vec![1, 2, 3],
        Some(String::from("foo")),
        Some(String::from("foo")),
        Some(String::from(r#"{"arg": "value"}"#)),
        Some(ChunkToolCall::default()),
        Some(ToolCall {
            id: String::from("foo"),
            r#type: String::from("function"),
            function: FunctionDefinition {
                name: String::from("foo"),
                arguments: serde_json::from_str(r#"{"arg": "value"}"#).unwrap(),
            },
        }),
        None,
    )]
    #[case(
        GenerationDataType::ToolCallArguments,
        vec![1, 2, 3],
        Some(String::from("foo")),
        Some(String::from("foo")),
        Some(String::from(r#"{"arg": "value""#)),
        Some(ChunkToolCall::default()),
        Some(ToolCall {
            id: String::from("foo"),
            r#type: String::from("function"),
            function: FunctionDefinition {
                name: String::from("foo"),
                arguments: serde_json::from_str(r#"{"arg": "value"}"#).unwrap(),
            },
        }),
        None,
    )]
    #[case(
        GenerationDataType::ToolCallName,
        vec![1, 2, 3],
        Some(String::from("foo")),
        Some(String::from("foo")),
        None,
        Some(ChunkToolCall::default()),
        Some(ToolCall {
            id: String::from("foo"),
            r#type: String::from("function"),
            function: FunctionDefinition {
                name: String::from("foo"),
                arguments: HashMap::new(),
            },
        }),
        None,
    )]
    #[case(
        GenerationDataType::ToolCallArguments,
        vec![1, 2, 3],
        Some(String::from("foo")),
        Some(String::from("foo")),
        Some(String::from("foo")),
        None,
        Some(ToolCall {
            id: String::from("foo"),
            r#type: String::from("function"),
            function: FunctionDefinition {
                name: String::from("foo"),
                arguments: HashMap::new(),
            },
        }),
        Some("Failed to parse model tool arguments into JSON".to_string()),
    )]
    fn test_finalize_tool_call(
        #[case] ongoing_gen_type: GenerationDataType,
        #[case] current_buffer_ids: Vec<u32>,
        #[case] current_tool_name: Option<String>,
        #[case] current_tool_id: Option<String>,
        #[case] current_tool_arguments: Option<String>,
        #[case] expected_chunk: Option<ChunkToolCall>,
        #[case] expected_tool_call: Option<ToolCall>,
        #[case] error_message: Option<String>,
    ) {
        let formatter = setup_mock_formatter(true, None, true);
        let mut tokenizer = MockTokenizer::new();
        let expected_function_name = expected_tool_call.as_ref().map(|v| v.function.name.clone());
        tokenizer
            .expect_process_multiple_token_stream()
            .returning(move |_, _| Ok(expected_function_name.clone()));
        let mut context = make_context(&tokenizer, formatter);
        context.current_tool_name = current_tool_name.unwrap_or_default();
        context.current_tool_id = current_tool_id.unwrap_or_default();
        context.current_tool_arguments = current_tool_arguments.unwrap_or_default();
        context.current_ids_buffer = current_buffer_ids;
        let mut supervisor = ToolCallingSupervisor {
            context: context,
            tool_call_parsing_state: StateEnum::None,
        };
        let result = supervisor.finalize_tool_call(&ongoing_gen_type);
        if let Some(error_message) = error_message {
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains(&error_message));
        } else {
            assert!(result.is_ok());
            let result = result.unwrap();
            assert_eq!(result.is_some(), expected_chunk.is_some());
            assert_eq!(supervisor.context.tool_calls.len(), 1);
            let last_tool_call = supervisor.context.tool_calls.last().unwrap();
            if expected_tool_call.is_some() {
                assert!(!last_tool_call.id.is_empty());
                assert_eq!(
                    last_tool_call.function.name,
                    expected_tool_call.clone().unwrap().function.name
                );
                assert_eq!(
                    last_tool_call.function.arguments,
                    expected_tool_call.unwrap().function.arguments
                );
            } else {
                assert!(last_tool_call.id.is_empty());
            }
        }
    }

    #[rstest]
    #[case(None, Some("name".to_string()))]
    #[case(Some("foo".to_string()), Some("name".to_string()))]
    fn test_emit_initial_tool_call_chunk(
        #[case] current_tool_id: Option<String>,
        #[case] current_tool_name: Option<String>,
    ) {
        let formatter = setup_mock_formatter(true, None, true);
        let tokenizer = MockTokenizer::new();
        let mut context = make_context(&tokenizer, formatter);
        context.current_tool_name = current_tool_name.clone().unwrap_or_default();
        context.current_tool_id = current_tool_id.clone().unwrap_or_default();
        let mut supervisor = ToolCallingSupervisor {
            context: context,
            tool_call_parsing_state: StateEnum::None,
        };
        let result = supervisor.emit_initial_tool_call_chunk();
        assert!(result.id.is_some());
        assert_eq!(result.index, 0);
        assert_eq!(result.r#type, Some("function".to_string()));
        assert_eq!(result.function.name, current_tool_name);
        assert!(result.function.arguments.is_empty());
        if let Some(tool_id) = current_tool_id {
            assert_eq!(result.id.unwrap(), tool_id);
        }
    }

    #[rstest]
    #[case("foo".to_string(), 0, true)]
    #[case("foo".to_string(), 1, true)]
    #[case("".to_string(), 0, false)]
    #[case("foo".to_string(), 4, false)]
    fn test_emit_arg_tool_call_chunk(
        #[case] current_tool_args: String,
        #[case] stream_args_pos: usize,
        #[case] result_is_some: bool,
    ) {
        let formatter = setup_mock_formatter(true, None, true);
        let tokenizer = MockTokenizer::new();
        let mut context = make_context(&tokenizer, formatter);
        context.current_tool_arguments = current_tool_args.clone();
        context.stream_args_pos = stream_args_pos;
        let mut supervisor = ToolCallingSupervisor {
            context: context,
            tool_call_parsing_state: StateEnum::None,
        };
        let result = supervisor.emit_arg_tool_call_chunk();
        assert_eq!(result.is_some(), result_is_some);
        if result_is_some {
            let result = result.unwrap();
            let mut tool_args = current_tool_args.clone();
            tool_args.drain(..stream_args_pos);
            assert_eq!(result.index, 0);
            assert!(result.id.is_none());
            assert!(result.r#type.is_none());
            assert!(result.function.name.is_none());
            assert_eq!(result.function.arguments, tool_args);
        }
    }

    #[test]
    fn test_reset() {
        let mut formatter = MockToolArgFormatter::new();
        formatter.expect_reset().times(1).returning(move || {});
        let tokenizer = MockTokenizer::new();
        let mut context = make_context(&tokenizer, Box::new(formatter));
        context.current_ids_buffer = vec![1, 2, 3];
        context.current_tool_name = "foo".to_string();
        context.current_tool_id = "bar".to_string();
        context.current_tool_arguments = "baz".to_string();
        context.stream_args_pos = 1;
        context.stream_ctx = StreamContext {
            ids: vec![4, 5, 6],
            prefix: "prefix".to_string(),
            prefix_index: 1,
        };
        let mut supervisor = ToolCallingSupervisor {
            context: context,
            tool_call_parsing_state: StateEnum::ArgumentsContent,
        };
        supervisor.reset();
        assert!(supervisor.context.current_ids_buffer.is_empty());
        assert!(supervisor.context.current_tool_name.is_empty());
        assert!(supervisor.context.current_tool_id.is_empty());
        assert!(supervisor.context.current_tool_arguments.is_empty());
        assert_eq!(supervisor.context.stream_args_pos, 0);
        assert_eq!(supervisor.tool_call_parsing_state, StateEnum::None);
        assert!(supervisor.context.stream_ctx.ids.is_empty());
        assert!(supervisor.context.stream_ctx.prefix.is_empty());
        assert_eq!(supervisor.context.stream_ctx.prefix_index, 0);
    }

    proptest! {
        #[test]
        fn test_set_tool_name(random_name in any::<String>()) {
            let name = random_name;
            let formatter = MockToolArgFormatter::new();
            let tokenizer = MockTokenizer::new();
            let context = make_context(&tokenizer, Box::new(formatter));
            let mut supervisor = ToolCallingSupervisor {
                context: context,
                tool_call_parsing_state: StateEnum::ArgumentsContent,
            };
            supervisor.context.set_tool_name(name.clone());
            let sanitized_name = supervisor.context.current_tool_name.clone();
            let original_len = name.len();

            // --- INVARIANT A: Validate Sanitization Rules ---
            for c in sanitized_name.chars() {
                prop_assert!(
                    c.is_alphanumeric() || c == '_',
                    "Found an invalid character '{}' inside output: '{}'", c, sanitized_name
                );
            }

            // --- INVARIANT B: Length Boundary Check ---
            prop_assert!(
                sanitized_name.len() <= original_len,
                "Output string somehow grew larger than original input!"
            );

            // --- INVARIANT C: Idempotency Verification ---
            // Passing the clean string back in shouldn't modify it further
            supervisor.context.set_tool_name(sanitized_name.clone());
            prop_assert_eq!(&supervisor.context.current_tool_name, &sanitized_name);
        }
    }
}
