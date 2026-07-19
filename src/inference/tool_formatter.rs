use crate::inference::PostSamplingConfig;
use crate::tokenizer::{StreamContext, Tokenizer};
#[cfg(test)]
use mockall::automock;
use std::any::Any;
use std::mem::take;

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ToolFormatStyle {
    XmlArgPairs,  // GLM-4 / Qwen 2.5 XML syntax
    #[default]
    EnclosedJson, // Qwen standard / Mistral JSON block
    PythonCall,   // Llama 3 custom script syntax
}

#[cfg_attr(test, automock)]
pub trait ToolArgFormatter {
    #[allow(clippy::needless_lifetimes)]
    fn build_tool_call_template<'a>(
        &self,
        tool_call_start_token_id: &u32,
        tool_name: Option<&'a str>,
        tokenizer: &dyn Tokenizer,
    ) -> anyhow::Result<Vec<u32>>;
    fn try_strip_name_prefix(&self, decoded_buffer: &mut String) -> bool;
    fn try_extract_function_name(
        &self,
        token_ids: &[u32],
        decoded_buffer: &mut String,
        tokenizer: &dyn Tokenizer,
        stream_ctx: &mut StreamContext,
    ) -> anyhow::Result<Option<String>>;
    fn try_strip_arguments_prefix(&self, decoded_buffer: &mut String) -> bool;
    fn fix_json(&mut self, decoded_buffer: &mut String, first_chunk: bool);
    fn format_args(
        &mut self,
        token_ids: &[u32],
        decoded_buffer: &mut String,
        tokenizer: &dyn Tokenizer,
        stream_ctx: &mut StreamContext,
    ) -> anyhow::Result<()>;
    fn reset(&mut self);
    fn as_any(&self) -> &dyn Any;
}

pub struct ToolArgFormatterBuilder<'a> {
    config: &'a PostSamplingConfig,
}

impl<'a> ToolArgFormatterBuilder<'a> {
    pub fn new(config: &'a PostSamplingConfig) -> Self {
        Self { config }
    }

    pub fn build(self) -> anyhow::Result<Box<dyn ToolArgFormatter>> {
        Ok(match self.config.tool_call_format_style {
            ToolFormatStyle::XmlArgPairs => Box::new(XmlArgPairsFormatter::new(
                self.config.arg_key_open_token_id.ok_or_else(|| {
                    anyhow::anyhow!("Missing arg_key_open_token_id for XmlArgPairsFormatter")
                })?,
                self.config.arg_key_close_token_id.ok_or_else(|| {
                    anyhow::anyhow!("Missing arg_key_close_token_id for XmlArgPairsFormatter")
                })?,
                self.config.arg_value_open_token_id.ok_or_else(|| {
                    anyhow::anyhow!("Missing arg_value_open_token_id for XmlArgPairsFormatter")
                })?,
                self.config.arg_value_close_token_id.ok_or_else(|| {
                    anyhow::anyhow!("Missing arg_value_close_token_id for XmlArgPairsFormatter")
                })?,
            )?),
            ToolFormatStyle::EnclosedJson => Box::new(EnclosedJsonFormatter::new()?),
            ToolFormatStyle::PythonCall => Box::new(PythonCallFormatter::new()?),
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
    json_arg_delimiter: String,
}

impl XmlArgPairsFormatter {
    pub fn new(
        arg_key_open_token_id: u32,
        arg_key_close_token_id: u32,
        arg_value_open_token_id: u32,
        arg_value_close_token_id: u32,
    ) -> anyhow::Result<Self> {
        let json_arg_start = r#"{""#.to_string();
        let json_arg_key_close = r#"": "#.to_string();
        let json_arg_value_wrapper = r#"""#.to_string();
        let json_arg_delimiter = r#", ""#.to_string();
        Ok(Self {
            arg_key_open_token_id,
            arg_key_close_token_id,
            arg_value_open_token_id,
            arg_value_close_token_id,
            json_arg_start,
            json_arg_key_close,
            json_arg_value_wrapper,
            json_arg_delimiter,
        })
    }
}

impl ToolArgFormatter for XmlArgPairsFormatter {
    fn build_tool_call_template(
        &self,
        tool_call_start_token_id: &u32,
        tool_name: Option<&str>,
        tokenizer: &dyn Tokenizer,
    ) -> anyhow::Result<Vec<u32>> {
        if let Some(name) = tool_name {
            let ids = tokenizer.encode(name, false)?;
            let mut result = Vec::with_capacity(ids.len() + 1);
            result.push(*tool_call_start_token_id);
            result.extend_from_slice(&ids);
            return Ok(result);
        }

        Ok(vec![*tool_call_start_token_id])
    }

    fn try_strip_name_prefix(&self, _decoded_buffer: &mut String) -> bool {
        true
    }

    fn try_extract_function_name(
        &self,
        token_ids: &[u32],
        decoded_buffer: &mut String,
        tokenizer: &dyn Tokenizer,
        stream_ctx: &mut StreamContext,
    ) -> anyhow::Result<Option<String>> {
        let mut name_and_remainder =
            token_ids.rsplitn(2, |item| item == &self.arg_key_open_token_id);
        let name_piece = name_and_remainder.next().unwrap();
        let remainder_option = name_and_remainder.next();

        match remainder_option {
            Some([]) => {
                if let Some(decoded_piece) =
                    tokenizer.process_multiple_token_stream(stream_ctx, name_piece)?
                {
                    decoded_buffer.push_str(&decoded_piece);
                } else {
                    stream_ctx.reset();
                }
                let name = take(decoded_buffer);
                Ok(Some(name))
            }
            Some(remainder) => {
                if let Some(decoded_piece) =
                    tokenizer.process_multiple_token_stream(stream_ctx, name_piece)?
                {
                    decoded_buffer.push_str(&decoded_piece);
                } else {
                    stream_ctx.reset();
                }
                let name = take(decoded_buffer);
                if let Some(remainder_str) =
                    tokenizer.process_multiple_token_stream(stream_ctx, remainder)?
                {
                    decoded_buffer.push_str(&remainder_str);
                }
                Ok(Some(name))
            }
            None => {
                if let Some(decoded_piece) =
                    tokenizer.process_multiple_token_stream(stream_ctx, name_piece)?
                {
                    decoded_buffer.push_str(&decoded_piece);
                }
                Ok(None)
            }
        }
    }

    fn try_strip_arguments_prefix(&self, _decoded_buffer: &mut String) -> bool {
        true
    }

    fn fix_json(&mut self, decoded_buffer: &mut String, first_chunk: bool) {
        if first_chunk && !decoded_buffer.is_empty() {
            decoded_buffer.insert_str(0, &self.json_arg_start);
        }
    }
    fn format_args(
        &mut self,
        token_ids: &[u32],
        decoded_buffer: &mut String,
        tokenizer: &dyn Tokenizer,
        stream_ctx: &mut StreamContext,
    ) -> anyhow::Result<()> {
        for id in token_ids {
            match *id {
                token_id if token_id == self.arg_key_open_token_id => {
                    decoded_buffer.push_str(&self.json_arg_delimiter);
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
    fn as_any(&self) -> &dyn Any {
        self
    }
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
    fn new() -> anyhow::Result<Self> {
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
    fn build_tool_call_template(
        &self,
        tool_call_start_token_id: &u32,
        tool_name: Option<&str>,
        tokenizer: &dyn Tokenizer,
    ) -> anyhow::Result<Vec<u32>> {
        let ids = if let Some(name) = tool_name {
            let template = format!(r#"{{"name":"{}""#, name);
            tokenizer.encode(&template, false)?
        } else {
            let template = r#"{"name":""#;
            tokenizer.encode(template, false)?
        };

        let mut result = Vec::with_capacity(ids.len() + 1);
        result.push(*tool_call_start_token_id);
        result.extend_from_slice(&ids);
        Ok(result)
    }

    fn try_strip_name_prefix(&self, decoded_buffer: &mut String) -> bool {
        let pref_start_idx_option = decoded_buffer.rfind(&self.name_prefix);
        match pref_start_idx_option {
            Some(pref_start_idx)
                if pref_start_idx + self.name_prefix.len() == decoded_buffer.len() =>
            {
                decoded_buffer.clear();
                true
            }
            Some(pref_start_idx) => {
                decoded_buffer.drain(0..pref_start_idx + self.name_prefix.len());
                true
            }
            None => false,
        }
    }

    fn try_extract_function_name(
        &self,
        token_ids: &[u32],
        decoded_buffer: &mut String,
        tokenizer: &dyn Tokenizer,
        stream_ctx: &mut StreamContext,
    ) -> anyhow::Result<Option<String>> {
        if let Some(decoded_piece) =
            tokenizer.process_multiple_token_stream(stream_ctx, token_ids)?
        {
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
            Some(pref_start_idx)
                if pref_start_idx + self.arguments_prefix.len() == decoded_buffer.len() =>
            {
                decoded_buffer.clear();
                true
            }
            Some(pref_start_idx) => {
                decoded_buffer.drain(0..pref_start_idx + self.arguments_prefix.len());
                true
            }
            None => false,
        }
    }

    fn fix_json(&mut self, decoded_buffer: &mut String, _first_chunk: bool) {
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
                    true
                }
                '}' | ']' => {
                    if self.bracket_depth > 0 {
                        self.bracket_depth -= 1;
                        true
                    } else {
                        false
                    }
                }
                _ => true,
            }
        });
    }

    fn format_args(
        &mut self,
        token_ids: &[u32],
        decoded_buffer: &mut String,
        tokenizer: &dyn Tokenizer,
        stream_ctx: &mut StreamContext,
    ) -> anyhow::Result<()> {
        if let Some(decoded_piece) =
            tokenizer.process_multiple_token_stream(stream_ctx, token_ids)?
        {
            decoded_buffer.push_str(&decoded_piece);
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.in_string = false;
        self.is_escaping = false;
        self.bracket_depth = 0;
    }

    fn as_any(&self) -> &dyn Any {
        self
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
    fn new() -> anyhow::Result<Self> {
        let name_suffix = r#"("#.to_string();

        Ok(Self {
            name_suffix,
            ..Default::default()
        })
    }
}

impl ToolArgFormatter for PythonCallFormatter {
    fn build_tool_call_template(
        &self,
        tool_call_start_token_id: &u32,
        tool_name: Option<&str>,
        tokenizer: &dyn Tokenizer,
    ) -> anyhow::Result<Vec<u32>> {
        if let Some(name) = tool_name {
            let ids = tokenizer.encode(&format!("[{}", name), false)?;
            let mut result = Vec::with_capacity(ids.len() + 1);
            result.push(*tool_call_start_token_id);
            result.extend_from_slice(&ids);
            return Ok(result);
        }

        Ok(vec![*tool_call_start_token_id])
    }

    fn try_strip_name_prefix(&self, _decoded_buffer: &mut String) -> bool {
        true
    }

    fn try_extract_function_name(
        &self,
        token_ids: &[u32],
        decoded_buffer: &mut String,
        tokenizer: &dyn Tokenizer,
        stream_ctx: &mut StreamContext,
    ) -> anyhow::Result<Option<String>> {
        if let Some(decoded_piece) =
            tokenizer.process_multiple_token_stream(stream_ctx, token_ids)?
        {
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
                    if c != ' ' {
                        // Clean out layout space drift
                        output.push(c);
                    }
                }
            }
        }

        decoded_buffer.clear();
        decoded_buffer.push_str(&output);
    }

    fn format_args(
        &mut self,
        token_ids: &[u32],
        decoded_buffer: &mut String,
        tokenizer: &dyn Tokenizer,
        stream_ctx: &mut StreamContext,
    ) -> anyhow::Result<()> {
        if let Some(decoded_piece) =
            tokenizer.process_multiple_token_stream(stream_ctx, token_ids)?
        {
            decoded_buffer.push_str(&decoded_piece);
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.in_string = false;
        self.is_escaping = false;
        self.bracket_depth = 0;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::MockTokenizer;
    use rstest::rstest;

    #[rstest]
    #[case(1, 2, 3, 4, vec![], "".to_string(), r#""#.to_string())]
    #[case(1, 2, 3, 4, vec![19], "".to_string(), r#"{"BYPASSED"#.to_string())]
    #[case(1, 2, 3, 4, vec![19, 2, 3, 19, 4], "".to_string(), r#"{"BYPASSED": "BYPASSED""#.to_string())]
    #[case(1, 2, 3, 4, vec![19, 2, 3, 19, 4, 1, 19, 19, 2, 3, 19, 19, 4], "".to_string(), r#"{"BYPASSED": "BYPASSED", "BYPASSEDBYPASSED": "BYPASSEDBYPASSED""#.to_string())]
    fn test_xml_arg_pairs_formatter_format_args(
        #[case] arg_key_open_token_id: u32,
        #[case] arg_key_close_token_id: u32,
        #[case] arg_value_open_token_id: u32,
        #[case] arg_value_close_token_id: u32,
        #[case] token_ids: Vec<u32>,
        #[case] decoded_buffer: String,
        #[case] decoded_buffer_expected: String,
    ) {
        let mut formatter = XmlArgPairsFormatter::new(
            arg_key_open_token_id,
            arg_key_close_token_id,
            arg_value_open_token_id,
            arg_value_close_token_id,
        )
        .unwrap();
        let mut decoded_buffer = decoded_buffer;
        let mut stream_ctx = StreamContext::with_capacity(10);
        let mut tokenizer = MockTokenizer::new();
        tokenizer
            .expect_process_token_stream()
            .returning(|_, _| Ok(Some("BYPASSED".to_string())));

        let _ = formatter.format_args(&token_ids, &mut decoded_buffer, &tokenizer, &mut stream_ctx);
        formatter.fix_json(&mut decoded_buffer, true);

        assert_eq!(decoded_buffer, decoded_buffer_expected);
    }

    #[rstest]
    #[case(true, "".to_string(), "".to_string())]
    #[case(false, "BYPASSED".to_string(), "BYPASSED".to_string())]
    #[case(true, "BYPASSED".to_string(), r#"{"BYPASSED"#.to_string())]
    fn test_xml_arg_pairs_formatter_fix_json(
        #[case] first_chunk: bool,
        #[case] decoded_buffer: String,
        #[case] decoded_buffer_expected: String,
    ) {
        let mut formatter = XmlArgPairsFormatter::new(1, 2, 3, 4).unwrap();
        let mut decoded_buffer = decoded_buffer;
        formatter.fix_json(&mut decoded_buffer, first_chunk);
        assert_eq!(decoded_buffer, decoded_buffer_expected);
    }

    #[rstest]
    #[case(9, None, vec![19], vec![9])]
    #[case(9, Some("Name"), vec![19], vec![9, 19])]
    #[case(9, Some("Name"), vec![19, 2, 3, 19, 4], vec![9, 19, 2, 3, 19, 4])]
    fn test_xml_arg_pairs_formatter_build_tool_call_template(
        #[case] tool_call_start_token_id: u32,
        #[case] tool_name: Option<&str>,
        #[case] tool_name_encoded: Vec<u32>,
        #[case] expected: Vec<u32>,
    ) {
        let formatter = XmlArgPairsFormatter::new(1, 2, 3, 4).unwrap();
        let mut tokenizer = MockTokenizer::new();
        tokenizer
            .expect_encode()
            .returning(move |_, _| Ok(tool_name_encoded.clone()));
        let result =
            formatter.build_tool_call_template(&tool_call_start_token_id, tool_name, &tokenizer);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(r#"{"name": ""#, "", true)]
    #[case(r#"{"name": "name"#, "name", true)]
    #[case(r#"prefix{"name": ""#, "", true)]
    #[case(r#"wdewdwe"#, "wdewdwe", false)]
    #[case(r#"prefix{"name": "name"#, "name", true)]
    fn test_enclosed_json_formatter_try_strip_name_prefix(
        #[case] decoded_buffer: &str,
        #[case] decoded_buffer_expected: &str,
        #[case] expected: bool,
    ) {
        let formatter = EnclosedJsonFormatter::new().unwrap();
        let mut decoded_buffer = decoded_buffer.to_string();
        let result = formatter.try_strip_name_prefix(&mut decoded_buffer);
        assert_eq!(result, expected);
        assert_eq!(decoded_buffer, decoded_buffer_expected);
    }

    #[rstest]
    #[case(r#"na"#, Some(r#"me""#.to_string()), "", Some("name".to_string()))]
    #[case(r#"name"#, None, r#"name"#, None)]
    #[case(r#"name"#, Some(r#"""#.to_string()), "", Some("name".to_string()))]
    #[case(r#"name"#, Some(r#""arg"#.to_string()), "arg", Some("name".to_string()))]
    #[case(r#"na"#, Some(r#"me"#.to_string()), "name", None)]
    fn test_enclosed_json_formatter_try_extract_function_name(
        #[case] decoded_buffer: &str,
        #[case] decoded_tokens: Option<String>,
        #[case] decoded_buffer_expected: &str,
        #[case] expected_name: Option<String>,
    ) {
        let formatter = EnclosedJsonFormatter::new().unwrap();
        let mut decoded_buffer = decoded_buffer.to_string();
        let mut tokenizer = MockTokenizer::new();
        tokenizer
            .expect_process_multiple_token_stream()
            .returning(move |_, _| Ok(decoded_tokens.clone()));
        let token_ids = vec![1, 2, 3];
        let result = formatter.try_extract_function_name(
            &token_ids,
            &mut decoded_buffer,
            &tokenizer,
            &mut StreamContext::with_capacity(10),
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result, expected_name);
        assert_eq!(decoded_buffer, decoded_buffer_expected);
    }

    #[rstest]
    #[case(r#""arguments":"#, "", true)]
    #[case(r#""arguments":suffix"#, "suffix", true)]
    #[case(r#""#, "", false)]
    #[case(r#"wdewdwe"#, "wdewdwe", false)]
    #[case(r#"prefix"arguments":suffix"#, "suffix", true)]
    fn test_enclosed_json_formatter_try_strip_arguments_prefix(
        #[case] decoded_buffer: &str,
        #[case] decoded_buffer_expected: &str,
        #[case] expected: bool,
    ) {
        let formatter = EnclosedJsonFormatter::new().unwrap();
        let mut decoded_buffer = decoded_buffer.to_string();
        let result = formatter.try_strip_arguments_prefix(&mut decoded_buffer);
        assert_eq!(result, expected);
        assert_eq!(decoded_buffer, decoded_buffer_expected);
    }

    #[rstest]
    #[case(r#""#, r#""#)]
    #[case(r#"}}"#, r#""#)]
    #[case(r#"{}}"#, r#"{}"#)]
    #[case(r#"{}"}""#, r#"{}"}""#)]
    #[case(r#"]{}}"#, r#"{}"#)]
    fn test_enclosed_json_formatter_fix_json(
        #[case] decoded_buffer: &str,
        #[case] decoded_buffer_expected: &str,
    ) {
        let mut formatter = EnclosedJsonFormatter::new().unwrap();
        let mut decoded_buffer = decoded_buffer.to_string();
        formatter.fix_json(&mut decoded_buffer, false);
        assert_eq!(decoded_buffer, decoded_buffer_expected);
    }

    #[test]
    fn test_enclosed_json_formatter_reset() {
        let mut formatter = EnclosedJsonFormatter::new().unwrap();
        formatter.in_string = true;
        formatter.is_escaping = true;
        formatter.bracket_depth = 1;
        formatter.reset();
        assert!(!formatter.in_string);
        assert!(!formatter.is_escaping);
        assert_eq!(formatter.bracket_depth, 0);
    }

    #[rstest]
    #[case(r#""#, r#""#, false)]
    #[case(r#"some"#, r#"{"some"#, true)]
    #[case(r#""}some"#, r#""}some"#, false)]
    #[case(r#"())"#, r#"()}"#, false)]
    #[case(r#"]some"#, r#"}"#, false)]
    #[case(r#"arg="value")"#, r#"{"arg": "value"}"#, true)]
    #[case(
        r#"arg1="value1", arg2="value2")"#,
        r#"{"arg1": "value1", "arg2": "value2"}"#,
        true
    )]
    fn test_python_call_formatter_fix_json(
        #[case] decoded_buffer: &str,
        #[case] decoded_buffer_expected: &str,
        #[case] first_chunk: bool,
    ) {
        let mut formatter = PythonCallFormatter::new().unwrap();
        let mut decoded_buffer = decoded_buffer.to_string();
        formatter.fix_json(&mut decoded_buffer, first_chunk);
        assert_eq!(decoded_buffer, decoded_buffer_expected);
    }

    #[rstest]
    #[case(r#"na"#, Some(r#"me("#.to_string()), "", Some("name".to_string()))]
    #[case(r#"name"#, None, r#"name"#, None)]
    #[case(r#"name"#, Some(r#"("#.to_string()), "", Some("name".to_string()))]
    #[case(r#"name"#, Some(r#"(arg"#.to_string()), "arg", Some("name".to_string()))]
    #[case(r#"na"#, Some(r#"me"#.to_string()), "name", None)]
    #[case(r#"."#, Some(r#",("#.to_string()), ".,", None)]
    fn test_python_call_formatter_try_extract_function_name(
        #[case] decoded_buffer: &str,
        #[case] decoded_tokens: Option<String>,
        #[case] decoded_buffer_expected: &str,
        #[case] expected_name: Option<String>,
    ) {
        let formatter = PythonCallFormatter::new().unwrap();
        let mut decoded_buffer = decoded_buffer.to_string();
        let mut tokenizer = MockTokenizer::new();
        tokenizer
            .expect_process_multiple_token_stream()
            .returning(move |_, _| Ok(decoded_tokens.clone()));
        let token_ids = vec![1, 2, 3];
        let result = formatter.try_extract_function_name(
            &token_ids,
            &mut decoded_buffer,
            &tokenizer,
            &mut StreamContext::with_capacity(10),
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result, expected_name);
        assert_eq!(decoded_buffer, decoded_buffer_expected);
    }

    #[rstest]
    #[case(r#"("#, "", false)]
    #[case(r#"some"#, "some", true)]
    #[case(r#"some("#, "some", true)]
    #[case(r#""#, "", false)]
    #[case(r#"some_some"#, "some_some", true)]
    fn test_python_call_formatter_try_strip_arguments_prefix(
        #[case] decoded_buffer: &str,
        #[case] decoded_buffer_expected: &str,
        #[case] expected: bool,
    ) {
        let formatter = PythonCallFormatter::new().unwrap();
        let mut decoded_buffer = decoded_buffer.to_string();
        let result = formatter.try_strip_arguments_prefix(&mut decoded_buffer);
        assert_eq!(result, expected);
        assert_eq!(decoded_buffer, decoded_buffer_expected);
    }

    #[rstest]
    #[case(ToolFormatStyle::XmlArgPairs, Some(1), Some(2), Some(3), Some(4), None)]
    #[case(ToolFormatStyle::EnclosedJson, None, None, None, None, None)]
    #[case(ToolFormatStyle::PythonCall, None, None, None, None, None)]
    #[case(ToolFormatStyle::XmlArgPairs, None, Some(2), Some(3), Some(4), Some("Missing arg_key_open_token_id for XmlArgPairsFormatter".to_string()))]
    fn test_tool_arg_formatter_build(
        #[case] tool_call_format_style: ToolFormatStyle,
        #[case] arg_key_open_token_id: Option<u32>,
        #[case] arg_key_close_token_id: Option<u32>,
        #[case] arg_value_open_token_id: Option<u32>,
        #[case] arg_value_close_token_id: Option<u32>,
        #[case] expected_error: Option<String>,
    ) {
        let config = PostSamplingConfig {
            tool_call_format_style,
            arg_key_open_token_id,
            arg_key_close_token_id,
            arg_value_open_token_id,
            arg_value_close_token_id,
            ..Default::default()
        };

        let builder = ToolArgFormatterBuilder::new(&config);
        let formatter_result = builder.build();
        match formatter_result {
            Ok(formatter) => match config.tool_call_format_style {
                ToolFormatStyle::XmlArgPairs => {
                    assert!(formatter.as_any().is::<XmlArgPairsFormatter>());
                }
                ToolFormatStyle::EnclosedJson => {
                    assert!(formatter.as_any().is::<EnclosedJsonFormatter>());
                }
                ToolFormatStyle::PythonCall => {
                    assert!(formatter.as_any().is::<PythonCallFormatter>());
                }
            },
            Err(e) => {
                assert!(e.to_string().contains(&expected_error.unwrap()));
            }
        }
    }

    #[test]
    fn test_python_call_formatter_reset() {
        let mut formatter = PythonCallFormatter::new().unwrap();
        formatter.in_string = true;
        formatter.is_escaping = true;
        formatter.bracket_depth = 1;
        formatter.reset();
        assert!(!formatter.in_string);
        assert!(!formatter.is_escaping);
        assert_eq!(formatter.bracket_depth, 0);
    }
}
