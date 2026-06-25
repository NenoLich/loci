use crate::error::LociError;
use crate::inference::{GenerationDataType, StreamFrame};
use colored::*;

pub fn stdout_callback(stream_frame: StreamFrame) -> Result<(), LociError> {
    let output_chunk = match stream_frame.output_type {
        GenerationDataType::DirectContent => stream_frame.output.green(),
        GenerationDataType::ToolCallName => {
            if let Some(tool_call_chunk) = stream_frame.tool_call_chunk {
                if let Some(name) = tool_call_chunk.function.name {
                    print!("\n Tool Call:\n");
                    format!("{}: ", name).blue().bold()
                } else {
                    "".normal()
                }
            } else {
                "".normal()
            }
        }
        GenerationDataType::ToolCallArguments => {
            if let Some(tool_call_chunk) = stream_frame.tool_call_chunk {
                tool_call_chunk.function.arguments.bright_cyan()
            } else {
                "".normal()
            }
        }
        GenerationDataType::Reasoning => stream_frame.output.bright_black().italic(),
    };

    print!("{}", output_chunk);
    use std::io::Write;
    std::io::stdout()
        .flush()
        .map_err(|e| LociError::Stream(e.to_string()))?;
    Ok(())
}
