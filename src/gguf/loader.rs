use crate::error::{LociContext, LociError};
use crate::gguf::{GgufHeaders, GgufInfo, GgufKVMeta, GgufTensorInfo, GgufType, GgufValue};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::{Mmap, MmapOptions};
use num_traits::FromPrimitive;
use std::fs::File;
use std::io::SeekFrom;
use std::io::{Cursor, Read, Seek, Write, stdout};
use std::path::Path;

const MAX_STRING_LEN: usize = 32 * 1024 * 1024; // 32 Megabytes
const MAX_ARRAY_ELEMENTS: usize = 5_000_000; // 5 Million elements

pub struct Loader;

impl Loader {
    #[tracing::instrument(level = "debug", skip_all)]
    pub fn load_gguf_info(
        path: impl AsRef<Path>,
        first_k_tensors: usize,
        verbose: bool,
    ) -> Result<GgufInfo, LociError> {
        let file = File::open(&path)?;
        let mmap: Mmap = unsafe { MmapOptions::new().map(&file).io_ctx("memory mapping")? };

        Self::load_gguf_info_from_bytes(&mmap, path, first_k_tensors, verbose)
    }

    fn load_gguf_info_from_bytes(
        bytes: &[u8],
        path: impl AsRef<Path>,
        first_k_tensors: usize,
        verbose: bool,
    ) -> Result<GgufInfo, LociError> {
        let mut cursor = Cursor::new(bytes);

        // 1. Load Headers
        let mut magic = [0u8; 4];
        cursor
            .read_exact(&mut magic)
            .io_ctx("reading magic bytes")?;
        if &magic != b"GGUF" {
            return Err(LociError::InvalidFileFormat(
                "GGUF file has magic bytes mismatch".into(),
            ));
        }

        let version = cursor
            .read_u32::<LittleEndian>()
            .io_ctx("reading version")?;
        let tensor_count = cursor
            .read_u64::<LittleEndian>()
            .io_ctx("reading tensor_count")?;
        let metadata_kv_count = cursor
            .read_u64::<LittleEndian>()
            .io_ctx("reading metadata_kv_count")?;

        let gguf_headers = GgufHeaders {
            path: path.as_ref().to_string_lossy().into_owned(),
            magic: String::from_utf8_lossy(&magic).into_owned(),
            version,
            tensor_count,
            metadata_kv_count,
        };

        // 2. Load Metadata
        let mut gguf_kv_meta_vec: Vec<GgufKVMeta> =
            Vec::with_capacity(gguf_headers.metadata_kv_count as usize);

        for _ in 0..gguf_headers.metadata_kv_count {
            let position = cursor.position();
            let key = Self::read_gguf_string(&mut cursor)
                .map_err(|e| e.with_meta_ctx("GGUF", "metadata key", position))?;
            let value_type = Self::read_gguf_type(&mut cursor)
                .map_err(|e| e.with_meta_ctx("GGUF", "metadata value type", position))?;
            let value = Self::get_gguf_value(&mut cursor, &value_type).map_err(|e| {
                e.with_meta_ctx(
                    "GGUF",
                    &format!("metadata value for key '{}'", key),
                    position,
                )
            })?;

            gguf_kv_meta_vec.push(GgufKVMeta {
                key,
                value_type,
                value,
            });
        }

        // 3. Load Tensor info
        let mut gguf_tensor_info: Vec<GgufTensorInfo> =
            Vec::with_capacity(gguf_headers.tensor_count as usize);

        for _ in 0..gguf_headers.tensor_count {
            let position = cursor.position();
            let name = Self::read_gguf_string(&mut cursor)
                .map_err(|e| e.with_meta_ctx("GGUF", "tensor name", position))?;
            let n_dims = Self::read_gguf_int32(&mut cursor).map_err(|e| {
                e.with_meta_ctx(
                    "GGUF",
                    &format!("tensor '{}' n_dims", name),
                    cursor.position(),
                )
            })?;
            let mut shapes: Vec<i64> = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let shape = Self::read_gguf_int64(&mut cursor).map_err(|e| {
                    e.with_meta_ctx(
                        "GGUF",
                        &format!("tensor '{}' shape", name),
                        cursor.position(),
                    )
                })?;
                shapes.push(shape);
            }

            let ggml_type = Self::read_gguf_int32(&mut cursor).map_err(|e| {
                e.with_meta_ctx(
                    "GGUF",
                    &format!("tensor '{}' ggml_type", name),
                    cursor.position(),
                )
            })?;
            let offset = Self::read_gguf_int64(&mut cursor).map_err(|e| {
                e.with_meta_ctx(
                    "GGUF",
                    &format!("tensor '{}' offset", name),
                    cursor.position(),
                )
            })?;
            gguf_tensor_info.push(GgufTensorInfo {
                name,
                n_dims,
                shapes,
                ggml_type,
                offset,
            });
        }

        // 4. Get the tensor offset start
        let alignment = Self::get_byte_alignment(&gguf_kv_meta_vec);
        let gguf_tensor_offset_start = Self::get_pos_with_byte_alignment(&mut cursor, alignment);

        if verbose {
            Self::print_gguf_headers(&mut stdout(), &gguf_headers)?;
            Self::print_gguf_kv_meta(&mut stdout(), &gguf_kv_meta_vec)?;
            Self::print_gguf_tensor_info(
                &mut stdout(),
                &gguf_tensor_info,
                first_k_tensors,
                gguf_headers.tensor_count,
            )?;
        }

        let gguf_info = GgufInfo {
            headers: gguf_headers,
            kv_meta: gguf_kv_meta_vec,
            tensor_info: gguf_tensor_info,
            tensor_offset_start: gguf_tensor_offset_start,
        };

        Ok(gguf_info)
    }

    fn print_gguf_headers<W: Write>(
        writer: &mut W,
        headers: &GgufHeaders,
    ) -> Result<(), LociError> {
        write!(
            writer,
            "\nloci Model Loader\n\
            ─────────────────────────────────\n\
            File: {}\n\
            Magic: {}\n\
            Version: {}\n\
            Tensor Count: {}\n\
            Metadata KV Count: {}\n\
            ─────────────────────────────────\n\
            Model file is valid GGUF format!\n",
            headers.path,
            headers.magic,
            headers.version,
            headers.tensor_count,
            headers.metadata_kv_count
        )
        .io_ctx("Printing GGUF headers")
    }

    fn print_gguf_kv_meta<W: Write>(
        writer: &mut W,
        kv_meta: &[GgufKVMeta],
    ) -> Result<(), LociError> {
        write!(
            writer,
            "\nMetadata:\n\
            ─────────────────────────────────\n"
        )
        .io_ctx("Printing GGUF metadata")?;

        for entry in kv_meta {
            if !matches!(entry.value.as_slice(), Some(array) if array.len() > 16) {
                writeln!(
                    writer,
                    "{}: {} = {}",
                    entry.key, entry.value_type, entry.value
                )
                .io_ctx("Printing GGUF metadata")?;
            } else {
                let slice = entry.value.as_slice().unwrap();
                let limit = 16.min(slice.len());
                writeln!(
                    writer,
                    "{}: {} = {:?}...[MORE THAN 16 ENTRIES]",
                    entry.key,
                    entry.value_type,
                    &slice[0..limit]
                )
                .io_ctx("Printing GGUF metadata")?;
            }
        }
        writeln!(writer, "─────────────────────────────────").io_ctx("Printing GGUF metadata")
    }

    fn print_gguf_tensor_info<W: Write>(
        writer: &mut W,
        tensor_info: &[GgufTensorInfo],
        first_k: usize,
        tensor_count: u64,
    ) -> Result<(), LociError> {
        let first_k_to_show = first_k.min(tensor_info.len());
        writeln!(
            writer,
            "Tensors (first {} of {}):",
            first_k_to_show, tensor_count
        )
        .io_ctx("Printing GGUF tensor info")?;
        writeln!(writer, "─────────────────────────────────")
            .io_ctx("Printing GGUF tensor info")?;
        for (i, item) in tensor_info.iter().enumerate().take(first_k_to_show) {
            writeln!(
                writer,
                "[{}] {} | n_dims: {} | shape: {:?} | type: {} | offset: {:#x}",
                i, item.name, item.n_dims, item.shapes, item.ggml_type, item.offset
            )
            .io_ctx("Printing GGUF tensor info")?;
        }
        writeln!(writer, "─────────────────────────────────").io_ctx("Printing GGUF tensor info")
    }

    fn read_gguf_uint8(cursor: &mut Cursor<&[u8]>) -> Result<u8, LociError> {
        cursor.read_u8().io_ctx("reading u8")
    }

    fn read_gguf_int8(cursor: &mut Cursor<&[u8]>) -> Result<i8, LociError> {
        cursor.read_i8().io_ctx("reading i8")
    }

    fn read_gguf_uint16(cursor: &mut Cursor<&[u8]>) -> Result<u16, LociError> {
        cursor.read_u16::<LittleEndian>().io_ctx("reading u16")
    }

    fn read_gguf_int16(cursor: &mut Cursor<&[u8]>) -> Result<i16, LociError> {
        cursor.read_i16::<LittleEndian>().io_ctx("reading i16")
    }

    fn read_gguf_uint32(cursor: &mut Cursor<&[u8]>) -> Result<u32, LociError> {
        cursor.read_u32::<LittleEndian>().io_ctx("reading u32")
    }

    fn read_gguf_int32(cursor: &mut Cursor<&[u8]>) -> Result<i32, LociError> {
        cursor.read_i32::<LittleEndian>().io_ctx("reading i32")
    }

    fn read_gguf_float32(cursor: &mut Cursor<&[u8]>) -> Result<f32, LociError> {
        cursor.read_f32::<LittleEndian>().io_ctx("reading f32")
    }

    fn read_gguf_uint64(cursor: &mut Cursor<&[u8]>) -> Result<u64, LociError> {
        cursor.read_u64::<LittleEndian>().io_ctx("reading u64")
    }

    fn read_gguf_int64(cursor: &mut Cursor<&[u8]>) -> Result<i64, LociError> {
        cursor.read_i64::<LittleEndian>().io_ctx("reading i64")
    }

    fn read_gguf_float64(cursor: &mut Cursor<&[u8]>) -> Result<f64, LociError> {
        cursor.read_f64::<LittleEndian>().io_ctx("reading f64")
    }

    fn read_gguf_bool(cursor: &mut Cursor<&[u8]>) -> Result<bool, LociError> {
        let value = cursor.read_i8().io_ctx("reading bool")? != 0;
        Ok(value)
    }

    fn read_gguf_array(cursor: &mut Cursor<&[u8]>) -> Result<Vec<GgufValue>, LociError> {
        let value_type = Self::read_gguf_type(cursor)?;
        let entries_num = cursor
            .read_u64::<LittleEndian>()
            .io_ctx("reading array length")? as usize;

        // Sanity limit check
        if entries_num > MAX_ARRAY_ELEMENTS {
            return Err(LociError::InvalidFileFormat(
                "Array length exceeds safety threshold".to_string(),
            ));
        }
        let mut values: Vec<GgufValue> = Vec::with_capacity(entries_num);
        for _ in 0..entries_num {
            let value = Self::get_gguf_value(cursor, &value_type)?;
            values.push(value);
        }
        Ok(values)
    }

    fn read_gguf_string(cursor: &mut Cursor<&[u8]>) -> Result<String, LociError> {
        let pos = cursor.position();
        let len = cursor
            .read_u64::<LittleEndian>()
            .io_ctx("reading string length")? as usize;

        // Sanity limit check
        if len > MAX_STRING_LEN {
            return Err(LociError::InvalidFileFormat(
                "String length exceeds safety threshold".to_string(),
            ));
        }

        let mut buffer = vec![0u8; len];
        cursor
            .read_exact(&mut buffer)
            .io_ctx("reading string bytes")?;
        String::from_utf8(buffer).map_err(|e| LociError::InvalidUtf8 {
            offset: pos,
            source: e,
        })
    }

    fn read_gguf_type(cursor: &mut Cursor<&[u8]>) -> Result<GgufType, LociError> {
        let gguf_type_n = cursor
            .read_i32::<LittleEndian>()
            .io_ctx("reading gguf type")?;
        GgufType::from_i32(gguf_type_n).ok_or(LociError::UnknownGgufType(gguf_type_n))
    }

    fn get_gguf_value(
        cursor: &mut Cursor<&[u8]>,
        value_type: &GgufType,
    ) -> Result<GgufValue, LociError> {
        let value = match value_type {
            GgufType::Uint8 => GgufValue::Uint8(Self::read_gguf_uint8(cursor)?),
            GgufType::Int8 => GgufValue::Int8(Self::read_gguf_int8(cursor)?),
            GgufType::Uint16 => GgufValue::Uint16(Self::read_gguf_uint16(cursor)?),
            GgufType::Int16 => GgufValue::Int16(Self::read_gguf_int16(cursor)?),
            GgufType::Uint32 => GgufValue::Uint32(Self::read_gguf_uint32(cursor)?),
            GgufType::Int32 => GgufValue::Int32(Self::read_gguf_int32(cursor)?),
            GgufType::Float32 => GgufValue::Float32(Self::read_gguf_float32(cursor)?),
            GgufType::Bool => GgufValue::Bool(Self::read_gguf_bool(cursor)?),
            GgufType::String => GgufValue::String(Self::read_gguf_string(cursor)?),
            GgufType::Array => GgufValue::Array(Self::read_gguf_array(cursor)?),
            GgufType::Uint64 => GgufValue::Uint64(Self::read_gguf_uint64(cursor)?),
            GgufType::Int64 => GgufValue::Int64(Self::read_gguf_int64(cursor)?),
            GgufType::Float64 => GgufValue::Float64(Self::read_gguf_float64(cursor)?),
        };
        Ok(value)
    }

    #[allow(dead_code)]
    fn skip_gguf_value(cursor: &mut Cursor<&[u8]>, value_type: &GgufType) -> Result<(), LociError> {
        match value_type {
            GgufType::Uint8 | GgufType::Int8 | GgufType::Bool => {
                cursor.seek(SeekFrom::Current(1)).io_ctx("skipping value")?;
            }
            GgufType::Uint16 | GgufType::Int16 => {
                cursor.seek(SeekFrom::Current(2)).io_ctx("skipping value")?;
            }
            GgufType::Uint32 | GgufType::Int32 | GgufType::Float32 => {
                cursor.seek(SeekFrom::Current(4)).io_ctx("skipping value")?;
            }
            GgufType::Uint64 | GgufType::Int64 | GgufType::Float64 => {
                cursor.seek(SeekFrom::Current(8)).io_ctx("skipping value")?;
            }
            GgufType::String => {
                let len = cursor
                    .read_u64::<LittleEndian>()
                    .io_ctx("reading string length for skip")?;
                cursor
                    .seek(SeekFrom::Current(len as i64))
                    .io_ctx("skipping string")?;
            }
            GgufType::Array => {
                let value_type = Self::read_gguf_type(cursor)?;
                let entries_num = cursor
                    .read_u64::<LittleEndian>()
                    .io_ctx("reading array length for skip")?;
                for _ in 0..entries_num {
                    Self::skip_gguf_value(cursor, &value_type)?;
                }
            }
        };
        Ok(())
    }

    fn get_byte_alignment(kv_meta: &[GgufKVMeta]) -> i64 {
        kv_meta
            .iter()
            .find(|entry| entry.key == "general.alignment")
            .and_then(|f| f.value.as_i64())
            .unwrap_or(32)
    }

    fn get_pos_with_byte_alignment(cursor: &mut Cursor<&[u8]>, alignment: i64) -> i64 {
        let alignment = alignment.max(1);
        let current_pos = cursor.position() as i64;
        (current_pos + (alignment - 1)) & !(alignment - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rstest::rstest;

    fn gguf_fixture() -> Vec<u8> {
        let mut fake_gguf = vec![];
        // Magic
        fake_gguf.extend_from_slice(b"GGUF");
        // Version
        fake_gguf.extend_from_slice(&3u32.to_le_bytes());
        // Tensor Count
        fake_gguf.extend_from_slice(&2u64.to_le_bytes());
        // Metadata KV Count
        fake_gguf.extend_from_slice(&1u64.to_le_bytes());
        // Metadata KV
        // Len string
        fake_gguf.extend_from_slice(&("general.alignment".len() as u64).to_le_bytes());
        // String
        fake_gguf.extend_from_slice(b"general.alignment");
        // Gguf type
        fake_gguf.extend_from_slice(&4i32.to_le_bytes());
        // Value
        fake_gguf.extend_from_slice(&32u32.to_le_bytes());
        // Tensor
        // Len string
        fake_gguf.extend_from_slice(&("tensor1".len() as u64).to_le_bytes());
        // String
        fake_gguf.extend_from_slice(b"tensor1");
        // N Dims
        fake_gguf.extend_from_slice(&2i32.to_le_bytes());
        // Shape
        fake_gguf.extend_from_slice(&2i64.to_le_bytes());
        fake_gguf.extend_from_slice(&3i64.to_le_bytes());
        // Ggml type
        fake_gguf.extend_from_slice(&3i32.to_le_bytes());
        // Offset
        fake_gguf.extend_from_slice(&0i64.to_le_bytes());
        // Tensor
        // Len string
        fake_gguf.extend_from_slice(&("tensor2".len() as u64).to_le_bytes());
        // String
        fake_gguf.extend_from_slice(b"tensor2");
        // N Dims
        fake_gguf.extend_from_slice(&2i32.to_le_bytes());
        // Shape
        fake_gguf.extend_from_slice(&2i64.to_le_bytes());
        fake_gguf.extend_from_slice(&3i64.to_le_bytes());
        // Ggml type
        fake_gguf.extend_from_slice(&3i32.to_le_bytes());
        // Offset
        fake_gguf.extend_from_slice(&0i64.to_le_bytes());

        fake_gguf
    }

    fn gguf_info_fixture() -> GgufInfo {
        GgufInfo {
            headers: GgufHeaders {
                path: String::from("test"),
                magic: "GGUF".to_string(),
                version: 3,
                tensor_count: 2,
                metadata_kv_count: 1,
            },
            kv_meta: vec![GgufKVMeta {
                key: "general.alignment".to_string(),
                value_type: GgufType::Uint32,
                value: GgufValue::Uint32(32),
            }],
            tensor_info: vec![
                GgufTensorInfo {
                    name: "tensor1".to_string(),
                    n_dims: 2,
                    shapes: vec![2, 3],
                    ggml_type: 3,
                    offset: 0,
                },
                GgufTensorInfo {
                    name: "tensor2".to_string(),
                    n_dims: 2,
                    shapes: vec![2, 3],
                    ggml_type: 3,
                    offset: 0,
                },
            ],
            tensor_offset_start: 160,
        }
    }

    #[rstest]
    #[case(gguf_fixture(), gguf_info_fixture())]
    #[case({
        let mut fake_gguf = gguf_fixture();
        fake_gguf[8] = 0x00;
        fake_gguf.truncate(57);
        fake_gguf
    }, {
        let mut fake_gguf_info = gguf_info_fixture();
        fake_gguf_info.headers.tensor_count = 0;
        fake_gguf_info.tensor_info = vec![];
        fake_gguf_info.tensor_offset_start = 64;
        fake_gguf_info
    })]
    fn test_load_gguf_info_from_bytes_success(
        #[case] bytes: Vec<u8>,
        #[case] expected_gguf_info: GgufInfo,
    ) {
        let loaded_gguf_info =
            Loader::load_gguf_info_from_bytes(&bytes, String::from("test"), 0, false)
                .expect("load_gguf_info_from_bytes expected to succeed but failed");
        assert_eq!(loaded_gguf_info, expected_gguf_info);
    }

    #[rstest]
    #[case(vec![], "reading magic bytes")]
    #[case({
        let mut fake_gguf = gguf_fixture();
        fake_gguf.insert(1, b'X');
        fake_gguf
    }, "GGUF file has magic bytes mismatch")]
    #[case({
        let mut fake_gguf = gguf_fixture();
        fake_gguf.truncate(4);
        fake_gguf
    }, "reading version")]
    #[case({
        let mut fake_gguf = gguf_fixture();
        fake_gguf.truncate(8);
        fake_gguf
    }, "reading tensor_count")]
    #[case({
        let mut fake_gguf = gguf_fixture();
        fake_gguf.truncate(16);
        fake_gguf
    }, "reading metadata_kv_count")]
    #[case({
        let mut fake_gguf = gguf_fixture();
        fake_gguf.truncate(24);
        fake_gguf
    }, "metadata key")]
    #[case({
        let mut fake_gguf = gguf_fixture();
        fake_gguf.truncate(49);
        fake_gguf
    }, "metadata value type")]
    #[case({
        let mut fake_gguf = gguf_fixture();
        fake_gguf.truncate(53);
        fake_gguf
    }, "metadata value for key 'general.alignment'")]
    #[case({
        let mut fake_gguf = gguf_fixture();
        fake_gguf.truncate(57);
        fake_gguf
    }, "tensor name")]
    #[case({
        let mut fake_gguf = gguf_fixture();
        fake_gguf.truncate(72);
        fake_gguf
    }, "tensor 'tensor1' n_dims")]
    #[case({
        let mut fake_gguf = gguf_fixture();
        fake_gguf.truncate(76);
        fake_gguf
    }, "tensor 'tensor1' shape")]
    #[case({
        let mut fake_gguf = gguf_fixture();
        fake_gguf.truncate(92);
        fake_gguf
    }, "tensor 'tensor1' ggml_type")]
    #[case({
        let mut fake_gguf = gguf_fixture();
        fake_gguf.truncate(96);
        fake_gguf
    }, "tensor 'tensor1' offset")]
    fn test_load_gguf_info_from_bytes_failure(
        #[case] bytes: Vec<u8>,
        #[case] expected_error_str: &str,
    ) {
        let result = Loader::load_gguf_info_from_bytes(&bytes, String::from("test"), 0, false)
            .expect_err("load_gguf_info_from_bytes expected to fail but succeeded");
        assert!(
            result.to_string().contains(expected_error_str),
            "load_gguf_info_from_bytes expected to fail with error containing '{}', but got '{}'",
            expected_error_str,
            result.to_string()
        );
    }

    #[test]
    fn test_print_gguf_headers() {
        let gguf_headers = GgufHeaders {
            magic: String::from("GGUF"),
            path: String::from("test"),
            version: 1,
            tensor_count: 2,
            metadata_kv_count: 1,
        };
        let mut buffer = vec![];
        Loader::print_gguf_headers(&mut buffer, &gguf_headers)
            .expect("print_gguf_headers expected to succeed but failed");
        let expected_output = String::from(
            "\nloci Model Loader\n\
            ─────────────────────────────────\n\
            File: test\n\
            Magic: GGUF\n\
            Version: 1\n\
            Tensor Count: 2\n\
            Metadata KV Count: 1\n\
            ─────────────────────────────────\n\
            Model file is valid GGUF format!\n",
        );
        assert_eq!(String::from_utf8(buffer).unwrap(), expected_output);
    }

    #[test]
    fn test_print_gguf_kv_meta() {
        let gguf_kv = GgufKVMeta {
            key: String::from("key"),
            value_type: GgufType::Uint32,
            value: GgufValue::Uint32(1),
        };
        let mut buffer = vec![];
        Loader::print_gguf_kv_meta(&mut buffer, &[gguf_kv])
            .expect("print_gguf_kv_meta expected to succeed but failed");
        let expected_output = String::from(
            "\nMetadata:\n\
            ─────────────────────────────────\n\
            key: Uint32 = 1\n\
            ─────────────────────────────────\n",
        );
        assert_eq!(String::from_utf8(buffer).unwrap(), expected_output);
    }

    #[rstest]
    #[case(
        0,
        String::from(
            "Tensors (first 0 of 2):\n\
            ─────────────────────────────────\n\
            ─────────────────────────────────\n"
        )
    )]
    #[case(
        1,
        String::from(
            "Tensors (first 1 of 2):\n\
            ─────────────────────────────────\n\
            [0] tensor1 | n_dims: 2 | shape: [1, 2] | type: 0 | offset: 0x0\n\
            ─────────────────────────────────\n"
        )
    )]
    #[case(
        2,
        String::from(
            "Tensors (first 2 of 2):\n\
            ─────────────────────────────────\n\
            [0] tensor1 | n_dims: 2 | shape: [1, 2] | type: 0 | offset: 0x0\n\
            [1] tensor2 | n_dims: 2 | shape: [3, 4] | type: 0 | offset: 0x0\n\
            ─────────────────────────────────\n"
        )
    )]
    #[case(
        4,
        String::from(
            "Tensors (first 2 of 2):\n\
            ─────────────────────────────────\n\
            [0] tensor1 | n_dims: 2 | shape: [1, 2] | type: 0 | offset: 0x0\n\
            [1] tensor2 | n_dims: 2 | shape: [3, 4] | type: 0 | offset: 0x0\n\
            ─────────────────────────────────\n"
        )
    )]
    fn test_print_gguf_tensor(#[case] first_k: usize, #[case] expected_output: String) {
        let gguf_tensor1 = GgufTensorInfo {
            name: String::from("tensor1"),
            n_dims: 2,
            shapes: vec![1, 2],
            ggml_type: 0,
            offset: 0,
        };
        let gguf_tensor2 = GgufTensorInfo {
            name: String::from("tensor2"),
            n_dims: 2,
            shapes: vec![3, 4],
            ggml_type: 0,
            offset: 0,
        };
        let mut buffer = vec![];
        Loader::print_gguf_tensor_info(
            &mut buffer,
            &[gguf_tensor1.clone(), gguf_tensor2.clone()],
            first_k,
            2,
        )
        .expect("print_gguf_tensor expected to succeed but failed");

        assert_eq!(String::from_utf8(buffer).unwrap(), expected_output);
    }

    #[rstest]
    #[case(vec![0x00], GgufType::Uint8, GgufValue::Uint8(0))]
    #[case(vec![0x01], GgufType::Int8, GgufValue::Int8(1))]
    #[case(vec![0x02, 0x03], GgufType::Uint16, GgufValue::Uint16(0x0302))]
    #[case(vec![0x04, 0x05], GgufType::Int16, GgufValue::Int16(0x0504))]
    #[case(vec![0x06, 0x07, 0x08, 0x09], GgufType::Uint32, GgufValue::Uint32(0x09080706))]
    #[case(vec![0x0A, 0x0B, 0x0C, 0x0D], GgufType::Int32, GgufValue::Int32(0x0D0C0B0A))]
    #[case(vec![0x0E, 0x0F, 0x10, 0x11], GgufType::Float32, GgufValue::Float32(f32::from_le_bytes([0x0E, 0x0F, 0x10, 0x11])))]
    #[case(vec![0x12], GgufType::Bool, GgufValue::Bool(true))]
    #[case(vec![0x00], GgufType::Bool, GgufValue::Bool(false))]
    // Fixed: 8-byte string lengths flipped to Little-Endian format
    #[case(vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], GgufType::String, GgufValue::String(String::from("")))]
    #[case(vec![0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, b'a', b'b', b'c', b'd'], GgufType::String, GgufValue::String(String::from("abcd")))]
    #[case(vec![0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, b'a', b'b', b'c', b'd', 0x00, b'e', b'f', b'g', b'h'], GgufType::String, GgufValue::String(String::from("abcd\0efgh")))]
    // Fixed: 8-byte integers flipped to Little-Endian format
    #[case(vec![0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00], GgufType::Uint64, GgufValue::Uint64(0x0001020304050607))]
    #[case(vec![0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08], GgufType::Int64, GgufValue::Int64(0x08090A0B0C0D0E0F))]
    #[case(vec![0x12, 0x13, 0x14, 0x15, 0x00, 0x00, 0x00, 0x00], GgufType::Float64, GgufValue::Float64(f64::from_le_bytes([0x12, 0x13, 0x14, 0x15, 0x00, 0x00, 0x00, 0x00])))]
    // Gguf arrays: 4-byte of gguf type, 8-byte of number of elements and elements
    #[case(vec![0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00], GgufType::Array, GgufValue::Array(vec![GgufValue::Uint32(0), GgufValue::Uint32(1), GgufValue::Uint32(2), GgufValue::Uint32(3)]))]
    fn test_get_gguf_value_success(
        #[case] bytes: Vec<u8>,
        #[case] value_type: GgufType,
        #[case] expected_value: GgufValue,
    ) {
        let mut cursor = Cursor::new(bytes.as_slice());
        let result = Loader::get_gguf_value(&mut cursor, &value_type)
            .expect("get_gguf_value expected to succeed but failed");
        assert_eq!(result, expected_value);
    }

    #[rstest]
    #[case(vec![], GgufType::Uint8, "reading u8")]
    #[case(vec![], GgufType::Int8, "reading i8")]
    #[case(vec![], GgufType::Uint16, "reading u16")]
    #[case(vec![], GgufType::Int16, "reading i16")]
    #[case(vec![], GgufType::Uint32, "reading u32")]
    #[case(vec![], GgufType::Int32, "reading i32")]
    #[case(vec![], GgufType::Float32, "reading f32")]
    #[case(vec![], GgufType::Bool, "reading bool")]
    // Gguf strings: 8-byte of string length and string
    #[case(vec![], GgufType::String, "reading string length")]
    #[case({
        let overlimitted_string_len = MAX_STRING_LEN + 1;
        let bytes = overlimitted_string_len.to_le_bytes().to_vec();
        bytes
    }, GgufType::String, "String length exceeds safety threshold")]
    #[case(vec![0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], GgufType::String, "reading string bytes")]
    // Gguf arrays: 4-byte of gguf type, 8-byte of number of elements and elements
    #[case(vec![], GgufType::Array, "reading gguf type")]
    #[case(vec![0x01, 0x00, 0x00, 0x00], GgufType::Array, "reading array length")]
    #[case({
        let mut bytes = vec![0x01, 0x00, 0x00, 0x00];
        let overlimitted_n_elements = MAX_ARRAY_ELEMENTS + 1;
        bytes.extend(overlimitted_n_elements.to_le_bytes());
        bytes
    }, GgufType::Array, "Array length exceeds safety threshold")]
    #[case(vec![], GgufType::Uint64, "reading u64")]
    #[case(vec![], GgufType::Int64, "reading i64")]
    #[case(vec![], GgufType::Float64, "reading f64")]
    fn test_get_gguf_value_failure(
        #[case] bytes: Vec<u8>,
        #[case] value_type: GgufType,
        #[case] expected_error_str: &str,
    ) {
        let mut cursor = Cursor::new(bytes.as_slice());
        let result = Loader::get_gguf_value(&mut cursor, &value_type)
            .expect_err("get_gguf_value expected to fail but succeeded");
        assert!(result.to_string().contains(expected_error_str));
    }

    #[test]
    fn test_read_gguf_type_unknown_gguf_type() {
        let bytes = vec![0x13, 0x00, 0x00, 0x00];
        let mut cursor = Cursor::new(bytes.as_slice());
        let result = Loader::read_gguf_type(&mut cursor)
            .expect_err("read_gguf_type expected to fail but succeeded");
        assert!(
            matches!(result, LociError::UnknownGgufType(..)),
            "read_gguf_type expected to fail with 'UnknownGgufType' but failed with '{result}'"
        );
    }

    #[rstest]
    #[case(vec![0x04, 0x00, 0x00, 0x00], GgufType::Uint8, 1)]
    #[case(vec![0x04, 0x00, 0x00, 0x00], GgufType::Int8, 1)]
    #[case(vec![0x04, 0x00, 0x00, 0x00], GgufType::Uint16, 2)]
    #[case(vec![0x04, 0x00, 0x00, 0x00], GgufType::Int16, 2)]
    #[case(vec![0x04, 0x00, 0x00, 0x00], GgufType::Uint32, 4)]
    #[case(vec![0x04, 0x00, 0x00, 0x00], GgufType::Int32, 4)]
    #[case(vec![0x04, 0x00, 0x00, 0x00], GgufType::Float32, 4)]
    #[case(vec![0x01, 0x00, 0x00, 0x00], GgufType::Bool, 1)]
    // Gguf strings: 8-byte of string length and string
    #[case(vec![0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], GgufType::String, 12)]
    // Gguf arrays: 4-byte of gguf type, 8-byte of number of elements and elements
    #[case(vec![0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00], GgufType::Array, 16)]
    #[case(vec![0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00], GgufType::Uint64, 8)]
    #[case(vec![0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00], GgufType::Int64, 8)]
    #[case(vec![0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00], GgufType::Float64, 8)]
    fn test_skip_gguf_value_success(
        #[case] bytes: Vec<u8>,
        #[case] value_type: GgufType,
        #[case] expected_position: u64,
    ) {
        let mut cursor = Cursor::new(bytes.as_slice());
        Loader::skip_gguf_value(&mut cursor, &value_type)
            .expect("skip_gguf_value expected to succeed but failed");
        assert_eq!(cursor.position(), expected_position);
    }

    #[rstest]
    // Gguf strings: 8-byte of string length and string
    #[case(vec![0x00], GgufType::String, "reading string length for skip")]
    fn test_skip_gguf_value_failure(
        #[case] bytes: Vec<u8>,
        #[case] value_type: GgufType,
        #[case] expected_error_str: &str,
    ) {
        let mut cursor = Cursor::new(bytes.as_slice());
        let result = Loader::skip_gguf_value(&mut cursor, &value_type)
            .expect_err("skip_gguf_value expected to fail but succeeded");
        assert!(
            result.to_string().contains(expected_error_str),
            "skip_gguf_value expected to fail with '{expected_error_str}' but failed with '{result}'"
        );
    }

    #[rstest]
    #[case(GgufKVMeta {
        key: "general.alignment".to_string(),
        value_type: GgufType::Uint32,
        value: GgufValue::Uint32(64),
    }, 64)]
    #[case(GgufKVMeta {
        key: "general.alignment".to_string(),
        value_type: GgufType::Uint32,
        value: GgufValue::Uint32(0),
    }, 0)]
    #[case(GgufKVMeta {
        key: "not.general.alignment".to_string(),
        value_type: GgufType::Uint32,
        value: GgufValue::Uint32(64),
    }, 32)]
    fn test_get_byte_alignment(#[case] kv_meta: GgufKVMeta, #[case] expected_alignment: i64) {
        let result = Loader::get_byte_alignment(&[kv_meta]);
        assert_eq!(result, expected_alignment);
    }

    proptest! {
        #[test]
        fn test_get_pos_with_byte_alignment_prop(
            start_pos in 0..10_000u64,
            alignment_power in 0..6
        ) {
            let alignment = 1i64 << alignment_power;
            let bytes = vec![0u8; 20_000];
            let mut cursor = Cursor::new(bytes.as_slice());
            cursor.set_position(start_pos);
            let current_pos_i64 = start_pos as i64;
            let result = Loader::get_pos_with_byte_alignment(&mut cursor, alignment);
            // --- INVARIANT A: Must be perfectly divisible by the alignment ---
            prop_assert_eq!(result % alignment, 0, "Result {} is not aligned to {}", result, alignment);

            // --- INVARIANT B: The aligned position cannot skip backwards ---
            prop_assert!(
                result >= current_pos_i64,
                "Aligned position ({}) cannot be behind the starting position ({})", result, current_pos_i64
            );

            // --- INVARIANT C: The step cannot be larger than the size of the alignment gap ---
            prop_assert!(
                (result - current_pos_i64) < alignment,
                "Alignment skipped an entire boundary block unnecessarily"
            );
        }
    }
}
