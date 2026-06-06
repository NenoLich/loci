use thiserror::Error;

pub trait LociContext<T> {
    fn io_ctx(self, ctx: &'static str) -> Result<T, LociError>;
}

impl<T> LociContext<T> for Result<T, std::io::Error> {
    fn io_ctx(self, ctx: &'static str) -> Result<T, LociError> {
        self.map_err(|e| LociError::IoWithContext { context: ctx, source: e })
    }
}

impl LociError {
    pub fn with_meta_ctx(self, file_type: &str, field: &str, offset: u64) -> Self {
        LociError::InvalidFileMetadata {
            file_type: file_type.into(),
            field: field.into(),
            offset,
            source: Box::new(self),
        }
    }
}

#[derive(Error, Debug)]
pub enum LociError {
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    #[error("Tokenizer build error: {reason}")]
    TokenizerBuild { reason: String },

    #[error("Tokenization failed: {source}")]
    Tokenization {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Invalid file format: {0}")]
    InvalidFileFormat(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Invalid {file_type} metadata for field '{field}' at offset {offset}")]
    InvalidFileMetadata {
        file_type: String,
        field: String,
        offset: u64,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("IO error at {context}: {source}")]
    IoWithContext {
        context: &'static str,
        #[source]
        source: std::io::Error,
    },

    #[error("Unknown GGUF type ID: {0}")]
    UnknownGgufType(i32),

    #[error("Invalid string encoding at offset {offset}: {source}")]
    InvalidUtf8 { offset: u64, source: std::string::FromUtf8Error },
}
