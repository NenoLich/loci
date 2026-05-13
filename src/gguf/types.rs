use num_derive::FromPrimitive;
use std::convert::From;
use std::fmt;

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

#[derive(FromPrimitive)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q40 = 2,
    Q41 = 3,
    Q50 = 6,
    Q51 = 7,
    Q80 = 8,
    Q81 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    Bf16 = 30,
    Tq10 = 34,
    Tq20 = 35,
    Mxfp4 = 39,
    Count = 40,
}

pub struct GgufInfo {
    pub headers: GgufHeaders,
    pub kv_meta: Vec<GgufKVMeta>,
    pub tensor_info: Vec<GgufTensorInfo>,
    pub tensor_offset_start: i64,
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

pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: i32,
    pub shapes: Vec<i64>,
    pub ggml_type: i32,
    pub offset: i64,
}

#[derive(Debug, Clone)]
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
        macro_rules!  display_value {
            ($($variant:ident),*) => {
                match self {
                    $(GgufValue::$variant(v) => write!(f, "{}", v),)*
                    GgufValue::Array(v) => f.debug_list().entries(v).finish(),
                }
            };
        }

        display_value!(
            Uint8, Int8, Uint16, Int16, Uint32, Int32, Float32, Bool, String, Uint64, Int64,
            Float64
        )
    }
}

impl GgufValue {
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            GgufValue::Uint8(v) => Some(usize::from(*v)),
            GgufValue::Uint16(v) => Some(usize::from(*v)),
            GgufValue::Uint32(v) => usize::try_from(*v).ok(),
            GgufValue::Int16(v) => usize::try_from(*v).ok(),
            GgufValue::Int32(v) => usize::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::Uint8(v) => Some(f32::from(*v)),
            GgufValue::Uint16(v) => Some(f32::from(*v)),
            GgufValue::Int16(v) => Some(f32::from(*v)),
            GgufValue::Float32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            GgufValue::Uint8(v) => Some(f64::from(*v)),
            GgufValue::Uint16(v) => Some(f64::from(*v)),
            GgufValue::Int16(v) => Some(f64::from(*v)),
            GgufValue::Float32(v) => Some(f64::from(*v)),
            GgufValue::Float64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::Uint8(v) => Some(u32::from(*v)),
            GgufValue::Uint16(v) => Some(u32::from(*v)),
            GgufValue::Uint32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            GgufValue::Uint8(v) => Some(i64::from(*v)),
            GgufValue::Uint16(v) => Some(i64::from(*v)),
            GgufValue::Uint32(v) => Some(i64::from(*v)),
            GgufValue::Int32(v) => Some(i64::from(*v)),
            _ => None,
        }
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

