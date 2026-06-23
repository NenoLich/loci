use crate::types::{ToolCall, ChunkToolCall, ChunkFunctionCall, FunctionDefinition, ToolChoice, ToolChoiceMode, SpecificToolChoice};
use crate::inference::{ToolArgFormatter, ToolArgFormatterBuilder, PostSamplingConfig, GenerationDataType, GenerationEvent};
use crate::tokenizer::{Tokenizer, StreamContext};
use uuid::Uuid;
use serde_json::Value;
use std::collections::HashMap;
use tracing::error;

#[derive(Debug)]
pub enum StateEnum {
    None,
    NamePrefix,
    NameContent,
    ArgumentsPrefix,
    ArgumentsContent,
}

impl StateEnum {
    fn next_state(
        &self,
        context: &mut ToolCallingSupervisorContext,
    ) -> anyhow::Result<StateEnum> {
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
        let decoded_tokens_option = context.tokenizer.process_multiple_token_stream(&mut context.stream_ctx, &context.current_ids_buffer)?;
        context.current_ids_buffer.clear();
        if let Some(decoded_tokens) = decoded_tokens_option {
            context.decoded_scratchpad.push_str(&decoded_tokens);

            if context.formatter.try_strip_name_prefix(&mut context.decoded_scratchpad) { 
                return NameContentState.next_state(context);
            }
        }

        Ok(StateEnum::NamePrefix)
    }
}

pub struct NameContentState;
impl ToolCallParsingState for NameContentState {
    fn next_state(&self, context: &mut ToolCallingSupervisorContext) -> anyhow::Result<StateEnum> {
        let tool_name_option = context.formatter.try_extract_function_name(&context.current_ids_buffer, &mut context.decoded_scratchpad, context.tokenizer, &mut context.stream_ctx)?;
        context.current_ids_buffer.clear();
        match tool_name_option {
            Some(tool_name) => {
                context.set_tool_name(tool_name);
                context.current_tool_id = Uuid::new_v4().to_string();

                ArgumentsPrefixState.next_state(context)
            },
            None => Ok(StateEnum::NameContent),
        }
    }
}

pub struct ArgumentsPrefixState;
impl ToolCallParsingState for ArgumentsPrefixState {
    fn next_state(&self, context: &mut ToolCallingSupervisorContext) -> anyhow::Result<StateEnum> {
        let decoded_tokens_option = context.tokenizer.process_multiple_token_stream(&mut context.stream_ctx, &context.current_ids_buffer)?;
        context.current_ids_buffer.clear();
        if let Some(decoded_tokens) = decoded_tokens_option {
            context.decoded_scratchpad.push_str(&decoded_tokens);

            if context.formatter.try_strip_arguments_prefix(&mut context.decoded_scratchpad) { 
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
    pub stream_ctx: StreamContext,                      // Streaming context for tool arguments
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
        self.current_tool_name.retain(|c| c.is_alphanumeric() || c == '_');
    }
    fn format_and_decode_args(&mut self) -> anyhow::Result<()> {
        self.formatter.format_args(&self.current_ids_buffer, &mut self.decoded_scratchpad, self.tokenizer, &mut self.stream_ctx)?;
        self.current_ids_buffer.clear();
        Ok(())
    }

    fn fix_and_update_args(&mut self) -> anyhow::Result<()> {
        self.formatter.fix_json(&mut self.decoded_scratchpad, self.current_tool_arguments.is_empty());
        self.current_tool_arguments.push_str(&self.decoded_scratchpad);
        self.decoded_scratchpad.clear();
        Ok(())
    }
}

pub struct ToolCallingSupervisor<'a> {
    pub context: ToolCallingSupervisorContext<'a>,
    pub tool_call_parsing_state: StateEnum,
}

impl<'a> ToolCallingSupervisor<'a> {
    pub fn new(supports_tool_calling: bool, config: &PostSamplingConfig, tokenizer: &'a dyn Tokenizer) -> anyhow::Result<Option<Self>> {
        if !supports_tool_calling {
            return Ok(None);
        }

        let formatter = ToolArgFormatterBuilder::new(config, tokenizer)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build tool argument formatter: {}", e))?;


        let Some(tool_call_start_token_id) = config.tool_call_start_token_id.clone() else {
            return Ok(None);
        };
        let Some(tool_call_end_token_id) = config.tool_call_end_token_id.clone() else {
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

    pub fn get_tool_choice_template(&self, tokenizer: &'a dyn Tokenizer, tool_choice: &ToolChoice) -> anyhow::Result<Option<Vec<u32>>> {
        match tool_choice {
            ToolChoice::Mode(ToolChoiceMode::Required) => {
                Ok(Some(self.context.formatter.build_tool_call_template(&self.context.tool_call_start_token_id, None, tokenizer)?))
            },
            ToolChoice::Specific(SpecificToolChoice { function, .. }) => {
                Ok(Some(self.context.formatter.build_tool_call_template(&self.context.tool_call_start_token_id, Some(&function.name), tokenizer)?))
            },
            _ => Ok(None),
        }
    }

    pub fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        Some(self.context.tool_calls.clone())
    }

    pub fn detect_tool_call_start(&mut self, token_ids: &[u32]) -> bool {
        token_ids.ends_with(std::slice::from_ref(&self.context.tool_call_start_token_id))
    }

    pub fn detect_tool_call_end(&self) -> bool {
        self.context.current_ids_buffer.ends_with(std::slice::from_ref(&self.context.tool_call_end_token_id))
    }

    pub fn finalize_tool_call(&mut self, ongoing_gen_type: &GenerationDataType) -> anyhow::Result<Option<ChunkToolCall>> {
        let chunk =match ongoing_gen_type {
            GenerationDataType::ToolCallName => {
                if self.context.current_tool_name.is_empty() {
                    // Corrected: Explicitly cast end sequence vector to a slice view reference
                    let suffix_slice = std::slice::from_ref(&self.context.tool_call_end_token_id);
                    let clean_ids = self.context.current_ids_buffer
                        .strip_suffix(suffix_slice)
                        .unwrap_or(&self.context.current_ids_buffer);
                        
                    if !clean_ids.is_empty() && let Some(tool_name) = self.context.tokenizer.process_multiple_token_stream(&mut self.context.stream_ctx, clean_ids)? {
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
            },
            
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

        let json_arguments: HashMap<String, Value> = serde_json::from_str(&self.context.current_tool_arguments)
            .map_err(|e| anyhow::anyhow!(
                "Failed to parse model tool arguments into JSON. Raw String: '{}'. Error: {}", 
                self.context.current_tool_arguments, e
            ))?;

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

    pub fn advance(&mut self, token_ids: &[u32], ongoing_gen_type: &GenerationDataType) -> anyhow::Result<GenerationEvent> {
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
                    self.tool_call_parsing_state = self.tool_call_parsing_state.next_state(&mut self.context)?;
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

    pub fn emit_tool_call_chunk_event(&mut self, ongoing_gen_type: &GenerationDataType) -> Option<GenerationEvent> {
        match ongoing_gen_type {
            GenerationDataType::ToolCallName => {
                if !self.context.current_tool_name.is_empty() {
                    Some(GenerationEvent::ToolCallNameChunk { chunk: self.emit_initial_tool_call_chunk() })
                } else {
                    None
                }
            }
            GenerationDataType::ToolCallArguments => {
                if let Some(chunk) = self.emit_arg_tool_call_chunk() {
                    Some(GenerationEvent::ToolCallArgumentsChunk { chunk })
                } else {
                    None  
                }
            }
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
        if self.context.current_tool_arguments.is_empty() || self.context.stream_args_pos >= self.context.current_tool_arguments.len() {
            return None;
        }
        let arguments = self.context.current_tool_arguments[self.context.stream_args_pos..].to_string();
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
