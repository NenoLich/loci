use num_derive::FromPrimitive;
use std::convert::From;
use std::fmt;

// GGUF Value Types
#[derive(FromPrimitive, Debug, PartialEq, Clone)]
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

#[allow(dead_code)]
#[derive(FromPrimitive, Debug, PartialEq)]
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

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct GgufInfo {
    pub headers: GgufHeaders,
    pub kv_meta: Vec<GgufKVMeta>,
    pub tensor_info: Vec<GgufTensorInfo>,
    pub tensor_offset_start: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufHeaders {
    pub path: String,
    pub magic: String,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufKVMeta {
    pub key: String,
    pub value_type: GgufType,
    pub value: GgufValue,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: i32,
    pub shapes: Vec<i64>,
    pub ggml_type: i32,
    pub offset: i64,
}

#[derive(Debug, Clone, PartialEq)]
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
                    GgufValue::Array(v) => {
                        write!(f, "[")?;
                        for (i, item) in v.iter().enumerate() {
                            if i > 0 {
                                write!(f, ", ")?;
                            }
                            write!(f, "{}", item)?;
                        }
                        write!(f, "]")
                    }
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

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::Uint8(v) => Some(u32::from(*v)),
            GgufValue::Uint16(v) => Some(u32::from(*v)),
            GgufValue::Uint32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self {
            GgufValue::Uint8(v) => Some(i32::from(*v)),
            GgufValue::Uint16(v) => Some(i32::from(*v)),
            GgufValue::Int8(v) => Some(i32::from(*v)),
            GgufValue::Int16(v) => Some(i32::from(*v)),
            GgufValue::Int32(v) => Some(*v),
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
        match self {
            GgufValue::Bool(v) => Some(*v),
            GgufValue::Uint8(0) => Some(false),
            GgufValue::Uint8(1) => Some(true),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        if let GgufValue::String(v) = self {
            Some(v)
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

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::FromPrimitive;
    use rstest::rstest;

    #[rstest]
    #[case(GgufValue::Uint8(42), Some(42))]
    #[case(GgufValue::Int32(100), Some(100))]
    #[case(GgufValue::Int32(-5), None)]
    #[case(GgufValue::Float32(3.14), None)]
    #[case(GgufValue::Bool(true), None)]
    #[case(GgufValue::String("hello".into()), None)]
    #[case(GgufValue::Array(vec![]), None)]
    fn test_as_usize(#[case] input: GgufValue, #[case] expected: Option<usize>) {
        assert_eq!(input.as_usize(), expected);
    }

    #[rstest]
    #[case(GgufValue::Uint8(42), Some(42.0))]
    #[case(GgufValue::Int32(100), None)]
    #[case(GgufValue::Int32(-5), None)]
    #[case(GgufValue::Float32(3.14), Some(3.14))]
    #[case(GgufValue::Bool(true), None)]
    #[case(GgufValue::String("hello".into()), None)]
    #[case(GgufValue::Array(vec![]), None)]
    fn test_as_f32(#[case] input: GgufValue, #[case] expected: Option<f32>) {
        assert_eq!(input.as_f32(), expected);
    }

    #[rstest]
    #[case(GgufValue::Uint8(42), Some(42))]
    #[case(GgufValue::Int32(100), None)]
    #[case(GgufValue::Int32(-5), None)]
    #[case(GgufValue::Float32(3.14), None)]
    #[case(GgufValue::Bool(true), None)]
    #[case(GgufValue::String("hello".into()), None)]
    #[case(GgufValue::Array(vec![]), None)]
    fn test_as_u32(#[case] input: GgufValue, #[case] expected: Option<u32>) {
        assert_eq!(input.as_u32(), expected);
    }

    #[rstest]
    #[case(GgufValue::Uint8(42), Some(42))]
    #[case(GgufValue::Int32(100), Some(100))]
    #[case(GgufValue::Int32(-5), Some(-5))]
    #[case(GgufValue::Float32(3.14), None)]
    #[case(GgufValue::Bool(true), None)]
    #[case(GgufValue::String("hello".into()), None)]
    #[case(GgufValue::Array(vec![]), None)]
    fn test_as_i32(#[case] input: GgufValue, #[case] expected: Option<i32>) {
        assert_eq!(input.as_i32(), expected);
    }

    #[rstest]
    #[case(GgufValue::Uint8(42), Some(42))]
    #[case(GgufValue::Int32(100), Some(100))]
    #[case(GgufValue::Int32(-5), Some(-5))]
    #[case(GgufValue::Float32(3.14), None)]
    #[case(GgufValue::Bool(true), None)]
    #[case(GgufValue::String("hello".into()), None)]
    #[case(GgufValue::Array(vec![]), None)]
    fn test_as_i64(#[case] input: GgufValue, #[case] expected: Option<i64>) {
        assert_eq!(input.as_i64(), expected);
    }

    #[rstest]
    #[case(GgufValue::Uint8(1), Some(true))]
    #[case(GgufValue::Uint8(0), Some(false))]
    #[case(GgufValue::Uint8(42), None)]
    #[case(GgufValue::Int32(100), None)]
    #[case(GgufValue::Int32(-5), None)]
    #[case(GgufValue::Float32(3.14), None)]
    #[case(GgufValue::Bool(true), Some(true))]
    #[case(GgufValue::String("hello".into()), None)]
    #[case(GgufValue::Array(vec![]), None)]
    fn test_as_bool(#[case] input: GgufValue, #[case] expected: Option<bool>) {
        assert_eq!(input.as_bool(), expected);
    }

    #[rstest]
    #[case(GgufValue::Uint8(42), None)]
    #[case(GgufValue::Int32(100), None)]
    #[case(GgufValue::Int32(-5), None)]
    #[case(GgufValue::Float32(3.14), None)]
    #[case(GgufValue::Bool(true), None)]
    #[case(GgufValue::String("hello".into()), Some("hello"))]
    #[case(GgufValue::Array(vec![]), None)]
    fn test_as_str(#[case] input: GgufValue, #[case] expected: Option<&str>) {
        assert_eq!(input.as_str(), expected);
    }

    #[rstest]
    #[case(GgufValue::Array(vec![
        GgufValue::Uint8(42),
        GgufValue::Int32(100),
        GgufValue::Int32(-5),
        GgufValue::Float32(3.14),
        GgufValue::Bool(true),
        GgufValue::String("hello".into()),
    ]), true, 6)]
    #[case(GgufValue::Array(vec![]), true, 0)]
    fn test_as_slice(#[case] input: GgufValue, #[case] is_some: bool, #[case] expected_len: usize) {
        let result = input.as_slice();
        assert_eq!(result.is_some(), is_some);
        let slice = result.unwrap_or(&[]);
        assert_eq!(slice.len(), expected_len);

        if expected_len == 6 {
            assert!(matches!(slice[0], GgufValue::Uint8(42)));
            assert!(matches!(slice[1], GgufValue::Int32(100)));
            assert!(matches!(slice[2], GgufValue::Int32(-5)));
            // Avoid direct float matching with matches! if precision is an issue,
            // but for hardcoded 3.14 exact layout it works:
            assert!(matches!(slice[3], GgufValue::Float32(f) if (f - 3.14).abs() < f32::EPSILON));
            assert!(matches!(slice[4], GgufValue::Bool(true)));
            assert!(matches!(&slice[5], GgufValue::String(s) if s == "hello"));
        }
    }
    #[rstest]
    #[case(GgufValue::Uint16(8), "8")]
    #[case(GgufValue::Int32(100), "100")]
    #[case(GgufValue::Int32(-5), "-5")]
    #[case(GgufValue::Float32(3.14), "3.14")]
    #[case(GgufValue::Bool(true), "true")]
    #[case(GgufValue::String("hello".into()), "hello")]
    #[case(GgufValue::Array(vec![]), "[]")]
    fn test_display_gguf_value(#[case] input: GgufValue, #[case] expected: &str) {
        assert_eq!(format!("{}", input), expected);
    }

    #[test]
    fn test_complex_array_snapshot() {
        let complex_nested_array = GgufValue::Array(vec![
            GgufValue::String("nested".into()),
            GgufValue::Array(vec![GgufValue::Uint8(1), GgufValue::Int16(-2)]),
        ]);
        insta::assert_snapshot!(format!("{}", complex_nested_array), @r###"[nested, [1, -2]]"###);
    }

    #[test]
    fn test_gguf_value_float32_snapshot() {
        insta::assert_snapshot!(format!("{}", GgufValue::Float32(3.14)), @"3.14");
    }

    #[rstest]
    #[case(0, GgmlType::F32)]
    #[case(1, GgmlType::F16)]
    #[case(2, GgmlType::Q40)]
    #[case(3, GgmlType::Q41)]
    #[case(6, GgmlType::Q50)]
    #[case(7, GgmlType::Q51)]
    #[case(8, GgmlType::Q80)]
    #[case(9, GgmlType::Q81)]
    #[case(10, GgmlType::Q2K)]
    #[case(11, GgmlType::Q3K)]
    #[case(12, GgmlType::Q4K)]
    #[case(13, GgmlType::Q5K)]
    #[case(14, GgmlType::Q6K)]
    #[case(15, GgmlType::Q8K)]
    #[case(16, GgmlType::Iq2Xxs)]
    #[case(17, GgmlType::Iq2Xs)]
    #[case(18, GgmlType::Iq3Xxs)]
    #[case(19, GgmlType::Iq1S)]
    #[case(20, GgmlType::Iq4Nl)]
    #[case(21, GgmlType::Iq3S)]
    #[case(22, GgmlType::Iq2S)]
    #[case(23, GgmlType::Iq4Xs)]
    #[case(24, GgmlType::I8)]
    #[case(25, GgmlType::I16)]
    #[case(26, GgmlType::I32)]
    #[case(27, GgmlType::I64)]
    #[case(28, GgmlType::F64)]
    #[case(29, GgmlType::Iq1M)]
    #[case(30, GgmlType::Bf16)]
    #[case(34, GgmlType::Tq10)]
    #[case(35, GgmlType::Tq20)]
    #[case(39, GgmlType::Mxfp4)]
    #[case(40, GgmlType::Count)]
    fn test_ggml_type_from_primitive(#[case] input: i32, #[case] expected: GgmlType) {
        assert_eq!(GgmlType::from_i32(input), Some(expected));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // A custom strategy to generate random GgufValues
    fn arb_gguf_value() -> impl Strategy<Value = GgufValue> {
        prop_oneof![
            any::<u8>().prop_map(GgufValue::Uint8),
            any::<i8>().prop_map(GgufValue::Int8),
            any::<u16>().prop_map(GgufValue::Uint16),
            any::<i16>().prop_map(GgufValue::Int16),
            any::<u32>().prop_map(GgufValue::Uint32),
            any::<i32>().prop_map(GgufValue::Int32),
            any::<f32>().prop_map(GgufValue::Float32),
            any::<bool>().prop_map(GgufValue::Bool),
            any::<String>().prop_map(GgufValue::String),
            any::<u64>().prop_map(GgufValue::Uint64),
            any::<i64>().prop_map(GgufValue::Int64),
            any::<f64>().prop_map(GgufValue::Float64),
            // Avoid deep recursion for arrays by using static definitions
            prop::collection::vec(any::<u8>(), 0..5)
                .prop_map(|v| GgufValue::Array(v.into_iter().map(GgufValue::Uint8).collect())),
        ]
    }

    proptest! {
        #[test]
        fn test_gguf_value_as_usize_invariants(v in arb_gguf_value()) {
            let result = v.as_usize();

            // Assert properties instead of explicit values
            match &v {
                GgufValue::Uint8(val)  => prop_assert_eq!(result, Some(*val as usize)),
                GgufValue::Uint16(val) => prop_assert_eq!(result, Some(*val as usize)),
                GgufValue::Uint32(val) => prop_assert_eq!(result, Some(*val as usize)),
                // Signed integers should only return Some if they are positive
                GgufValue::Int16(val)  => {
                    if *val >= 0 { prop_assert_eq!(result, Some(*val as usize)); }
                    else { prop_assert!(result.is_none()); }
                },
                GgufValue::Int32(val)  => {
                    if *val >= 0 { prop_assert_eq!(result, Some(*val as usize)); }
                    else { prop_assert!(result.is_none()); }
                },

                // Everything else MUST return None
                _ => prop_assert!(result.is_none()),
            }
        }
    }
}
