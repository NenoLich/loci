use crate::inference::PostSamplingConfig;
use crate::tokenizer::{TokenizerService, StreamContext};
use std::mem::take;

#[derive(Debug, Clone)]
pub enum ToolFormatStyle {
    XmlArgPairs,    // GLM-4 / Qwen 2.5 XML syntax
    EnclosedJson,   // Qwen standard / Mistral JSON block
    PythonCall,     // Llama 3 custom script syntax
}

pub trait ToolArgFormatter {
    fn build_tool_call_template(&self, tool_call_start_token_id: &u32, tool_name: Option<&str>, tokenizer: &TokenizerService) -> anyhow::Result<Vec<u32>>;
    fn try_strip_name_prefix(&self, decoded_buffer: &mut String) -> bool;
    fn try_extract_function_name(&self, token_ids: &[u32], decoded_buffer: &mut String, tokenizer: &TokenizerService, stream_ctx: &mut StreamContext) -> anyhow::Result<Option<String>>;
    fn try_strip_arguments_prefix(&self, decoded_buffer: &mut String) -> bool;
    fn fix_json(&mut self, decoded_buffer: &mut String, first_chunk: bool);
    fn format_args(&mut self, token_ids: &[u32], decoded_buffer: &mut String, tokenizer: &TokenizerService, stream_ctx: &mut StreamContext) -> anyhow::Result<()>;
    fn reset(&mut self);
}

pub struct ToolArgFormatterBuilder<'a> {
    config: &'a PostSamplingConfig,
    tokenizer: &'a TokenizerService,
}

impl<'a> ToolArgFormatterBuilder<'a> {
    pub fn new(config: &'a PostSamplingConfig, tokenizer: &'a TokenizerService) -> Self {
        Self { config , tokenizer }
    }

    pub fn build(self) -> anyhow::Result<Box<dyn ToolArgFormatter>> {
        Ok(match self.config.tool_call_format_style {
            ToolFormatStyle::XmlArgPairs => Box::new(XmlArgPairsFormatter::new(
                self.config.arg_key_open_token_id.ok_or_else(
                || anyhow::anyhow!("Missing arg_key_open_token_id for XmlArgPairsFormatter")
                )?, 
                self.config.arg_key_close_token_id.ok_or_else(
                || anyhow::anyhow!("Missing arg_key_close_token_id for XmlArgPairsFormatter")
                )?,
                self.config.arg_value_open_token_id.ok_or_else(
                || anyhow::anyhow!("Missing arg_value_open_token_id for XmlArgPairsFormatter")
                )?,
                self.config.arg_value_close_token_id.ok_or_else(
                || anyhow::anyhow!("Missing arg_value_close_token_id for XmlArgPairsFormatter")
                )?,
                &self.tokenizer,
            )?),
            ToolFormatStyle::EnclosedJson => Box::new(EnclosedJsonFormatter::new(&self.tokenizer)?),
            ToolFormatStyle::PythonCall => Box::new(PythonCallFormatter::new(&self.tokenizer)?),
        })
    }
}

pub struct XmlArgPairsFormatter {
    arg_key_open_token_id: u32,
    arg_key_close_token_id: u32,
    arg_value_open_token_id: u32,
    arg_value_close_token_id: u32,
    json_arg_start: String,
    json_arg_key_close: String,
    json_arg_value_wrapper: String,
    json_arg_delimeter: String,
}

impl XmlArgPairsFormatter {
    pub fn new(
        arg_key_open_token_id: u32,
        arg_key_close_token_id: u32,
        arg_value_open_token_id: u32,
        arg_value_close_token_id: u32,
        tokenizer: &TokenizerService
    ) -> anyhow::Result<Self> {
        let json_arg_start = r#"{""#.to_string();
        let json_arg_key_close = r#"": "#.to_string();
        let json_arg_value_wrapper = r#"""#.to_string();
        let json_arg_delimeter = r#", ""#.to_string();
        Ok(Self {
            arg_key_open_token_id,
            arg_key_close_token_id,
            arg_value_open_token_id,
            arg_value_close_token_id,
            json_arg_start,
            json_arg_key_close,
            json_arg_value_wrapper,
            json_arg_delimeter,
        })
    }
}

impl ToolArgFormatter for XmlArgPairsFormatter {
    fn build_tool_call_template(&self, tool_call_start_token_id: &u32, tool_name: Option<&str>, tokenizer: &TokenizerService) -> anyhow::Result<Vec<u32>> {
        if let Some(name) = tool_name {
            let encoding = tokenizer.encode(name, false)?;
            let mut result = Vec::with_capacity(encoding.len() + 1);
            result.push(*tool_call_start_token_id);
            result.extend_from_slice(encoding.get_ids());
            return Ok(result);
        } 
        
        Ok(vec![*tool_call_start_token_id])
    }

    fn try_strip_name_prefix(&self, decoded_buffer: &mut String) -> bool {
        true
    }

    fn try_extract_function_name(&self, token_ids: &[u32], decoded_buffer: &mut String, tokenizer: &TokenizerService, stream_ctx: &mut StreamContext) -> anyhow::Result<Option<String>> {
        let mut name_and_remainder = token_ids.rsplitn(2, |item| item == &self.arg_key_open_token_id);
        let name_piece = name_and_remainder.next().unwrap();
        let remainder_option = name_and_remainder.next();
        
        match remainder_option {
            Some([]) => {
                if let Some(decoded_piece) = tokenizer.process_multiple_token_stream(stream_ctx,name_piece)? {
                    decoded_buffer.push_str(&decoded_piece);
                } else {
                    stream_ctx.reset();
                }
                let name = take(decoded_buffer);   
                Ok(Some(name))
  
            }
            Some(remainder) => {
                if let Some(decoded_piece) = tokenizer.process_multiple_token_stream(stream_ctx,name_piece)? {
                    decoded_buffer.push_str(&decoded_piece);
                } else {
                    stream_ctx.reset();
                }
                let name = take(decoded_buffer); 
                if let Some(remainder_str) = tokenizer.process_multiple_token_stream(stream_ctx, remainder)? {
                    decoded_buffer.push_str(&remainder_str);
                }      
                Ok(Some(name))
            }
            None => {
                if let Some(decoded_piece) = tokenizer.process_multiple_token_stream(stream_ctx,name_piece)? {
                    decoded_buffer.push_str(&decoded_piece);
                }
                Ok(None)
            }
        }

    }

    fn try_strip_arguments_prefix(&self, decoded_buffer: &mut String) -> bool {
        true
    }

    fn fix_json(&mut self, decoded_buffer: &mut String, first_chunk: bool) {
        if first_chunk && !decoded_buffer.is_empty() {
            decoded_buffer.insert_str(0, r#"{{""#);
        }
    }
    fn format_args(&mut self, token_ids: &[u32], decoded_buffer: &mut String, tokenizer: &TokenizerService, stream_ctx: &mut StreamContext) -> anyhow::Result<()> {
        for id in token_ids {
            match *id {
                token_id if token_id == self.arg_key_open_token_id => {
                    decoded_buffer.push_str(&self.json_arg_delimeter);
                }
                token_id if token_id == self.arg_key_close_token_id => {
                    decoded_buffer.push_str(&self.json_arg_key_close);
                }
                token_id if token_id == self.arg_value_open_token_id => {
                    decoded_buffer.push_str(&self.json_arg_value_wrapper);
                }
                token_id if token_id == self.arg_value_close_token_id => {
                    decoded_buffer.push_str(&self.json_arg_value_wrapper);
                }
                _ => {
                    if let Some(decoded_piece) = tokenizer.process_token_stream(stream_ctx, *id)? {
                        decoded_buffer.push_str(&decoded_piece);
                    }
                }
            }
        }
        Ok(())
    }

    fn reset(&mut self) {}
}

#[derive(Default)]
pub struct EnclosedJsonFormatter {
    name_prefix: String,
    name_suffix: String,
    arguments_prefix: String,
    in_string: bool,
    is_escaping: bool,
    bracket_depth: usize,
}

impl EnclosedJsonFormatter {
    fn new(tokenizer: &TokenizerService) -> anyhow::Result<Self> {
        let name_prefix = r#"{"name": ""#.to_string();
        let name_suffix = r#"""#.to_string();
        let arguments_prefix = r#""arguments":"#.to_string();
    
        Ok(Self {
            name_prefix,
            name_suffix,
            arguments_prefix,
            ..Default::default()
        })
    }
}

impl ToolArgFormatter for EnclosedJsonFormatter {
    fn build_tool_call_template(&self, tool_call_start_token_id: &u32, tool_name: Option<&str>, tokenizer: &TokenizerService) -> anyhow::Result<Vec<u32>> {
        let encoding = if let Some(name) = tool_name {
            let template = format!(r#"{{"name":"{}""#, name);
            tokenizer.encode(&template, false)?     
        } else {
            let template = r#"{"name":""#;
            tokenizer.encode(template, false)?
        };

        let mut result = Vec::with_capacity(encoding.len() + 1);
        let tool_name_ids = encoding.get_ids();
        result.push(*tool_call_start_token_id);
        result.extend_from_slice(tool_name_ids);
        Ok(result)
    }
    
    fn try_strip_name_prefix(&self, decoded_buffer: &mut String) -> bool {
        let pref_start_idx_option = decoded_buffer.rfind(&self.name_prefix);
        match pref_start_idx_option {
            Some(pref_start_idx) if pref_start_idx + self.name_prefix.len() == decoded_buffer.len() => {
                decoded_buffer.clear();
                true
            }
            Some(pref_start_idx) => {
                decoded_buffer.drain(0..pref_start_idx + self.name_prefix.len());
                true
            }
            None => {
                false
            }
        }
    }

    fn try_extract_function_name(&self, token_ids: &[u32], decoded_buffer: &mut String, tokenizer: &TokenizerService, stream_ctx: &mut StreamContext) -> anyhow::Result<Option<String>> {
        if let Some(decoded_piece) = tokenizer.process_multiple_token_stream(stream_ctx, token_ids)? {
            match decoded_piece.rsplit_once(&self.name_suffix) {
                Some((name_piece, "")) => {
                    decoded_buffer.push_str(name_piece);
                    let name = take(decoded_buffer);          
                    Ok(Some(name))
                }
                Some((name_piece, remainder_str)) => {
                    decoded_buffer.push_str(name_piece);
                    let name = take(decoded_buffer); 
                    decoded_buffer.push_str(remainder_str);         
                    Ok(Some(name))
                }
                None => {
                    decoded_buffer.push_str(&decoded_piece);
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        } 
    }

    fn try_strip_arguments_prefix(&self, decoded_buffer: &mut String) -> bool {
        let pref_start_idx_option = decoded_buffer.rfind(&self.arguments_prefix);
        match pref_start_idx_option {
            Some(pref_start_idx) if pref_start_idx + self.arguments_prefix.len() == decoded_buffer.len() => {
                decoded_buffer.clear();
                true
            }
            Some(pref_start_idx) => {
                decoded_buffer.drain(0..pref_start_idx + self.arguments_prefix.len());
                true
            }
            None => {
                false
            }
        }
    }

    fn fix_json(&mut self, decoded_buffer: &mut String, first_chunk: bool) {
        decoded_buffer.retain(|c| {
                if self.is_escaping {
                    self.is_escaping = false;
                    return true;
                }
                if c == '\\' {
                    self.is_escaping = true;
                    return true;
                }
                if c == '"' {
                    self.in_string = !self.in_string;
                    return true;
                }
                if self.in_string {
                    return true;
                }
                match c {
                    '{' | '[' => {
                        self.bracket_depth += 1;
                        return true;
                    }
                    '}' | ']' => {
                        if self.bracket_depth > 0 {
                            self.bracket_depth -= 1;
                            return true;
                        } else {
                            return false;
                        }
                    }
                    _ => return true,
                }
                false
            });
    }

    fn format_args(&mut self, token_ids: &[u32], decoded_buffer: &mut String, tokenizer: &TokenizerService, stream_ctx: &mut StreamContext) -> anyhow::Result<()> {
        if let Some(decoded_piece) = tokenizer.process_multiple_token_stream(stream_ctx, token_ids)? {
            decoded_buffer.push_str(&decoded_piece);
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.in_string = false;
        self.is_escaping = false;
        self.bracket_depth = 0;
    }
}

#[derive(Default)]
pub struct PythonCallFormatter {
    name_suffix: String,
    in_string: bool,
    is_escaping: bool,
    bracket_depth: usize,
    envelope_closed: bool, 
}

impl PythonCallFormatter {
    fn new(tokenizer: &TokenizerService) -> anyhow::Result<Self> {
        let name_suffix = r#"("#.to_string();
    
        Ok(Self {
            name_suffix,
            ..Default::default()
        })
    }
}

impl ToolArgFormatter for PythonCallFormatter {
    fn build_tool_call_template(&self, tool_call_start_token_id: &u32, tool_name: Option<&str>, tokenizer: &TokenizerService) -> anyhow::Result<Vec<u32>> {
        if let Some(name) = tool_name {
            let encoding = tokenizer.encode(&format!("[{}", name), false)?;
            let mut result = Vec::with_capacity(encoding.len() + 1);
            result.push(*tool_call_start_token_id);
            result.extend_from_slice(encoding.get_ids());
            return Ok(result);
        } 
        
        Ok(vec![*tool_call_start_token_id])
    }

    fn try_strip_name_prefix(&self, decoded_buffer: &mut String) -> bool {
        true
    }

    fn try_extract_function_name(&self, token_ids: &[u32], decoded_buffer: &mut String, tokenizer: &TokenizerService, stream_ctx: &mut StreamContext) -> anyhow::Result<Option<String>> {
        if let Some(decoded_piece) = tokenizer.process_multiple_token_stream(stream_ctx, token_ids)? {
            match decoded_piece.rsplit_once(&self.name_suffix) {
                Some((name_piece, "")) => {
                    decoded_buffer.push_str(name_piece);
                    if decoded_buffer.chars().any(|c| c.is_alphanumeric()) {
                        let name = take(decoded_buffer);           
                        Ok(Some(name))
                    } else {
                        Ok(None)
                    }
                    
                }
                Some((name_piece, remainder_str)) => {
                    decoded_buffer.push_str(name_piece);      
                    if decoded_buffer.chars().any(|c| c.is_alphanumeric()) {
                        let name = take(decoded_buffer); 
                        decoded_buffer.push_str(remainder_str);          
                        Ok(Some(name))
                    } else {
                        decoded_buffer.push_str(remainder_str);
                        Ok(None)
                    }
                }
                None => {
                    decoded_buffer.push_str(&decoded_piece);
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        } 
    }

    fn try_strip_arguments_prefix(&self, decoded_buffer: &mut String) -> bool {
        decoded_buffer.retain(|c| c.is_alphanumeric() || c == '_');     
        !decoded_buffer.is_empty()
    }

    fn fix_json(&mut self, decoded_buffer: &mut String, first_chunk: bool) {
        if self.envelope_closed {
            decoded_buffer.clear();
            return; 
        }

        let mut output = String::with_capacity(decoded_buffer.len() + 4);
        if first_chunk && !decoded_buffer.is_empty() {
            output.push('{');
            output.push('"');
        }

        for c in decoded_buffer.chars() {
            if self.is_escaping {
                self.is_escaping = false;
                output.push(c);
                continue;
            }
            if c == '\\' {
                self.is_escaping = true;
                output.push(c);
                continue;
            }
            if c == '"' {
                self.in_string = !self.in_string;
                output.push(c);
                continue;
            }
            if self.in_string {
                output.push(c);
                continue;
            }

            match c {
                '(' | '[' => {
                    self.bracket_depth += 1;
                    output.push(c);
                }
                ')' | ']' => {
                    if self.bracket_depth > 0 {
                        self.bracket_depth -= 1;
                        output.push(c);
                    } else {
                        output.push('}');
                        self.envelope_closed = true;
                        break;
                    }
                }
                '=' => {
                    output.push('"');
                    output.push(':');
                    output.push(' ');
                }
                ',' => {
                    output.push(',');
                    output.push(' ');
                    output.push('"'); // Prep the next incoming parameter key string quote natively
                }
                _ => {
                    if c != ' ' { // Clean out layout space drift
                        output.push(c);
                    }
                }
            }
        }

        decoded_buffer.clear();
        decoded_buffer.push_str(&output);
    }

    fn format_args(&mut self, token_ids: &[u32], decoded_buffer: &mut String, tokenizer: &TokenizerService, stream_ctx: &mut StreamContext) -> anyhow::Result<()> {
        if let Some(decoded_piece) = tokenizer.process_multiple_token_stream(stream_ctx, token_ids)? {
            decoded_buffer.push_str(&decoded_piece);
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.in_string = false;
        self.is_escaping = false;
        self.bracket_depth = 0;
    }
}