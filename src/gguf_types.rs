use num_derive::FromPrimitive;
use std::fmt;
use std::convert::From;

// GGUF Value Types
#[derive(FromPrimitive)]
#[repr(u32)]
pub enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl fmt::Display for GgufType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GgufType::Uint8 => write!(f, "Uint8"),
            GgufType::Int8 => write!(f, "Int8"),
            GgufType::Uint16 => write!(f, "Uint16"),
            GgufType::Int16 => write!(f, "Int16"),
            GgufType::Uint32 => write!(f, "Uint32"),
            GgufType::Int32 => write!(f, "Int32"),
            GgufType::Float32 => write!(f, "Float32"),
            GgufType::Bool => write!(f, "Bool"),
            GgufType::String => write!(f, "String"),
            GgufType::Array => write!(f, "Array"),
            GgufType::Uint64 => write!(f, "Uint64"),
            GgufType::Int64 => write!(f, "Int64"),
            GgufType::Float64 => write!(f, "Float64"),
        }
    }
}

pub struct GgufInfo {
    pub headers: GgufHeaders,
    pub kv_meta: Vec<GgufKVMeta>,
}

pub struct GgufHeaders {
    pub path: String,
    pub magic: String,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

pub struct GgufKVMeta {
    pub key: String,
    pub value_type: GgufType,
    pub value: GgufValue,
}

pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>), 
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl fmt::Display for GgufValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GgufValue::Uint8(v) => write!(f, "{}", v),
            GgufValue::Int8(v) => write!(f, "{}", v),
            GgufValue::Uint16(v) => write!(f, "{}", v),
            GgufValue::Int16(v) => write!(f, "{}", v),
            GgufValue::Uint32(v) => write!(f, "{}", v),
            GgufValue::Int32(v) => write!(f, "{}", v),
            GgufValue::Float32(v) => write!(f, "{}", v),
            GgufValue::Bool(v) => write!(f, "{}", v),
            GgufValue::String(v) => write!(f, "{}", v),
            GgufValue::Array(v) => {
                    write!(f, "[")?;
                    for (i, val) in v.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", val)?;
                    }
                    write!(f, "]")
                },
            GgufValue::Uint64(v) => write!(f, "{}", v),
            GgufValue::Int64(v) => write!(f, "{}", v),
            GgufValue::Float64(v) => write!(f, "{}", v),
        }
    }
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        return match self {
            GgufValue::Uint8(v) => Some(*v as u32),
            GgufValue::Uint16(v) => Some(*v as u32),
            GgufValue::Uint32(v) => Some(*v),
            _ => None,
        };
    }

    pub fn as_bool(&self) -> Option<bool> {
        if let GgufValue::Bool(v) = self {
            Some(v.to_owned())
        } else {
            None
        }
    }

    pub fn as_string(&self) -> Option<String> {
        if let GgufValue::String(v) = self {
            Some(v.to_owned())
        } else {
            None
        }
    }

    pub fn as_slice(&self) -> Option<&[GgufValue]> {
        if let GgufValue::Array(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

pub struct GGUFTokenizerConfig {
    pub model_type: Option<String>, // from tokenizer.ggml.model
    pub tokens: Option<Vec<String>>,
    pub merges: Option<Vec<String>>,
    pub json_config: Option<String>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub add_bos: bool,
    pub add_eos: bool,
}

impl From<&[GgufKVMeta]> for GGUFTokenizerConfig {
    fn from(metadata: &[GgufKVMeta]) -> Self {
        let mut json_config = None;
        let mut model_type = None;
        let mut tokens = None;
        let mut merges = None;
        let mut bos_token_id = None;
        let mut eos_token_id = None;
        let mut add_bos = false;
        let mut add_eos = false;

        for kv_meta in metadata {
            match kv_meta.key.as_str() {
                "tokenizer.ggml.hf_json" => 
                    json_config = kv_meta.value.as_string(),
                "tokenizer.ggml.model" => 
                    model_type = kv_meta.value.as_string(),
                "tokenizer.ggml.tokens" => 
                    tokens = kv_meta.value.as_slice()
                        .and_then(|slice| {
                            slice.iter()
                                .filter_map(|v:&GgufValue| v.as_string().map(Some))
                                .collect::<Option<Vec<String>>>()
                        }),
                "tokenizer.ggml.merges" => 
                    merges = kv_meta.value.as_slice()
                        .and_then(|slice| {
                            slice.iter()
                                .filter_map(|v:&GgufValue| v.as_string().map(Some))
                                .collect::<Option<Vec<String>>>()
                        }),
                "tokenizer.ggml.bos_token_id" => bos_token_id = kv_meta.value.as_u32(),
                "tokenizer.ggml.eos_token_id" => eos_token_id = kv_meta.value.as_u32(),
                "tokenizer.ggml.add_bos_token" => 
                    add_bos = kv_meta.value.as_bool().is_some_and(|v| v),
                "tokenizer.ggml.add_eos_token" => 
                    add_eos = kv_meta.value.as_bool().is_some_and(|v| v),
                _ => {},
            }
        }

        Self {
            json_config,
            model_type,
            tokens,
            merges,
            bos_token_id,
            eos_token_id,
            add_bos,
            add_eos,
        }
    }
}