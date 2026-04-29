use crate::gguf_types::{GgufHeaders, GgufInfo, GgufKVMeta, GgufTensorInfo, GgufType, GgufValue};
use crate::error::{LociContext, LociError};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::{Mmap, MmapOptions};
use num_traits::FromPrimitive;
use std::fs::File;
use std::io::SeekFrom;
use std::io::{Cursor, Read, Seek};
use std::path::Path;

pub struct Loader;

impl Loader {
    pub fn load_gguf_info(
        path: impl AsRef<Path>,
        first_k_tensors: usize,
        verbose: bool,
    ) -> Result<GgufInfo, LociError> {
        let file = File::open(&path)?;
        let mmap: Mmap = unsafe {
            MmapOptions::new().map(&file).io_ctx("memory mapping")?
        };
        let mut cursor = Cursor::new(&mmap[..]);

        // 1. Load Headers
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic).map_err(|e| {
            LociError::IoWithContext { context: "reading magic bytes", source: e }
        })?;
        if &magic != b"GGUF" {
            return Err(LociError::InvalidFileFormat(
                "GGUF file has magic bytes mismatch".into(),
            ));
        }

        let version = cursor.read_u32::<LittleEndian>()
            .map_err(|e| LociError::IoWithContext { context: "reading version", source: e })?;

        let tensor_count = cursor.read_u64::<LittleEndian>()
            .map_err(|e| LociError::IoWithContext { context: "reading tensor_count", source: e })?;

        let metadata_kv_count = cursor.read_u64::<LittleEndian>()
            .map_err(|e| LociError::IoWithContext { context: "reading metadata_kv_count", source: e })?;

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
            let key = Self::read_gguf_string(&mut cursor).map_err(|e| {
                e.with_meta_ctx("GGUF", "metadata key", position)
            })?;
            let value_type = Self::read_gguf_type(&mut cursor).map_err(|e| {
                LociError::InvalidFileMetadata {
                    file_type: "GGUF".into(),
                    field: "metadata value type".into(),
                    offset: position,
                    source: Box::new(e),
                }
            })?;
            let value = Self::get_gguf_value(&mut cursor, &value_type).map_err(|e| {
                LociError::InvalidFileMetadata {
                    file_type: "GGUF".into(),
                    field: format!("metadata value for key '{}'", key).into(),
                    offset: position,
                    source: Box::new(e),
                }
            })?;

            gguf_kv_meta_vec.push(GgufKVMeta { key, value_type, value });
        }

        // 3. Load Tensor info
        let mut gguf_tensor_info: Vec<GgufTensorInfo> =
            Vec::with_capacity(gguf_headers.tensor_count as usize);

        for _ in 0..gguf_headers.tensor_count {
            let position = cursor.position();
            let name = Self::read_gguf_string(&mut cursor).map_err(|e| {
                e.with_meta_ctx("GGUF", "tensor name", position)
            })?;
            let n_dims = Self::read_gguf_int32(&mut cursor).map_err(|e| {
                LociError::InvalidFileMetadata {
                    file_type: "GGUF".into(),
                    field: format!("tensor '{}' n_dims", name).into(),
                    offset: cursor.position(),
                    source: Box::new(e),
                }
            })?;
            let mut shapes: Vec<i64> = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let shape = Self::read_gguf_int64(&mut cursor).map_err(|e| {
                    LociError::InvalidFileMetadata {
                        file_type: "GGUF".into(),
                        field: format!("tensor '{}' shape", name).into(),
                        offset: cursor.position(),
                        source: Box::new(e),
                    }
                })?;
                shapes.push(shape);
            }

            let ggml_type = Self::read_gguf_int32(&mut cursor).map_err(|e| {
                LociError::InvalidFileMetadata {
                    file_type: "GGUF".into(),
                    field: format!("tensor '{}' ggml_type", name).into(),
                    offset: cursor.position(),
                    source: Box::new(e),
                }
            })?;
            let offset = Self::read_gguf_int64(&mut cursor).map_err(|e| {
                LociError::InvalidFileMetadata {
                    file_type: "GGUF".into(),
                    field: format!("tensor '{}' offset", name).into(),
                    offset: cursor.position(),
                    source: Box::new(e),
                }
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
            Self::print_gguf_headers(&gguf_headers);
            Self::print_gguf_kv_meta(&gguf_kv_meta_vec);
            Self::print_gguf_tensor_info(&gguf_tensor_info, first_k_tensors, gguf_headers.tensor_count);
        }

        let gguf_info = GgufInfo {
            headers: gguf_headers,
            kv_meta: gguf_kv_meta_vec,
            tensor_info: gguf_tensor_info,
            tensor_offset_start: gguf_tensor_offset_start,
        };

        Ok(gguf_info)
    }

    fn print_gguf_headers(headers: &GgufHeaders) {
        println!("loci Model Loader");
        println!("─────────────────────────────────");
        println!("File: {}", headers.path);
        println!("Magic: {}", headers.magic);
        println!("Version: {}", headers.version);
        println!("Tensor Count: {}", headers.tensor_count);
        println!("Metadata KV Count: {}", headers.metadata_kv_count);
        println!("─────────────────────────────────");
        println!("Model file is valid GGUF format!");
    }

    fn print_gguf_kv_meta(kv_meta: &[GgufKVMeta]) {
        println!("\nMetadata:");
        println!("─────────────────────────────────");
        for entry in kv_meta {
            if !matches!(entry.value.as_slice(), Some(array) if array.len() > 16) {
                println!("{}: {} = {}", entry.key, entry.value_type, entry.value);
            } else {
                println!("{}: {} = [MORE THEN 16 ENTRIES]", entry.key, entry.value_type);
            }
        }
        println!("─────────────────────────────────");
    }

    fn print_gguf_tensor_info(tensor_info: &[GgufTensorInfo], first_k: usize, tensor_count: u64) {
        let first_k_to_show = first_k.min(tensor_info.len());
        println!("Tensors (first {} of {}):", first_k_to_show, tensor_count);
        println!("─────────────────────────────────");
        for i in 0..first_k_to_show {
            println!(
                "[{}] {} | n_dims: {} | shape: {:?} | type: {} | offset: {:#x}",
                i, tensor_info[i].name, tensor_info[i].n_dims,
                tensor_info[i].shapes, tensor_info[i].ggml_type, tensor_info[i].offset
            );
        }
        println!("─────────────────────────────────");
    }

    fn read_gguf_uint8(cursor: &mut Cursor<&[u8]>) -> Result<u8, LociError> {
        cursor.read_u8().map_err(|e| LociError::IoWithContext { context: "reading u8", source: e })
    }

    fn read_gguf_int8(cursor: &mut Cursor<&[u8]>) -> Result<i8, LociError> {
        cursor.read_i8().map_err(|e| LociError::IoWithContext { context: "reading i8", source: e })
    }

    fn read_gguf_uint16(cursor: &mut Cursor<&[u8]>) -> Result<u16, LociError> {
        cursor.read_u16::<LittleEndian>()
            .map_err(|e| LociError::IoWithContext { context: "reading u16", source: e })
    }

    fn read_gguf_int16(cursor: &mut Cursor<&[u8]>) -> Result<i16, LociError> {
        cursor.read_i16::<LittleEndian>()
            .map_err(|e| LociError::IoWithContext { context: "reading i16", source: e })
    }

    fn read_gguf_uint32(cursor: &mut Cursor<&[u8]>) -> Result<u32, LociError> {
        cursor.read_u32::<LittleEndian>()
            .map_err(|e| LociError::IoWithContext { context: "reading u32", source: e })
    }

    fn read_gguf_int32(cursor: &mut Cursor<&[u8]>) -> Result<i32, LociError> {
        cursor.read_i32::<LittleEndian>()
            .map_err(|e| LociError::IoWithContext { context: "reading i32", source: e })
    }

    fn read_gguf_float32(cursor: &mut Cursor<&[u8]>) -> Result<f32, LociError> {
        cursor.read_f32::<LittleEndian>()
            .map_err(|e| LociError::IoWithContext { context: "reading f32", source: e })
    }

    fn read_gguf_uint64(cursor: &mut Cursor<&[u8]>) -> Result<u64, LociError> {
        cursor.read_u64::<LittleEndian>()
            .map_err(|e| LociError::IoWithContext { context: "reading u64", source: e })
    }

    fn read_gguf_int64(cursor: &mut Cursor<&[u8]>) -> Result<i64, LociError> {
        cursor.read_i64::<LittleEndian>()
            .map_err(|e| LociError::IoWithContext { context: "reading i64", source: e })
    }

    fn read_gguf_float64(cursor: &mut Cursor<&[u8]>) -> Result<f64, LociError> {
        cursor.read_f64::<LittleEndian>()
            .map_err(|e| LociError::IoWithContext { context: "reading f64", source: e })
    }

    fn read_gguf_bool(cursor: &mut Cursor<&[u8]>) -> Result<bool, LociError> {
        let value = cursor.read_i8()
            .map_err(|e| LociError::IoWithContext { context: "reading bool", source: e })? != 0;
        Ok(value)
    }

    fn read_gguf_array(cursor: &mut Cursor<&[u8]>) -> Result<Vec<GgufValue>, LociError> {
        let value_type = Self::read_gguf_type(cursor)?;
        let entries_num = cursor.read_u64::<LittleEndian>()
            .map_err(|e| LociError::IoWithContext { context: "reading array length", source: e })? as usize;
        let mut values: Vec<GgufValue> = Vec::with_capacity(entries_num);
        for _ in 0..entries_num {
            let value = Self::get_gguf_value(cursor, &value_type)?;
            values.push(value);
        }
        Ok(values)
    }

    fn read_gguf_string(cursor: &mut Cursor<&[u8]>) -> Result<String, LociError> {
        let pos = cursor.position();
        let len = cursor.read_u64::<LittleEndian>()
            .map_err(|e| LociError::IoWithContext { context: "reading string length", source: e })? as usize;
        let mut buffer = vec![0u8; len];
        cursor.read_exact(&mut buffer)
            .map_err(|e| LociError::IoWithContext { context: "reading string bytes", source: e })?;
        String::from_utf8(buffer).map_err(|e| LociError::InvalidUtf8 { offset: pos, source: e })
    }

    fn read_gguf_type(cursor: &mut Cursor<&[u8]>) -> Result<GgufType, LociError> {
        let gguf_type_n = cursor.read_i32::<LittleEndian>()
            .map_err(|e| LociError::IoWithContext { context: "reading gguf type", source: e })?;
        GgufType::from_i32(gguf_type_n).ok_or(LociError::UnknownGgufType(gguf_type_n))
    }

    fn get_gguf_value(cursor: &mut Cursor<&[u8]>, value_type: &GgufType) -> Result<GgufValue, LociError> {
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

    fn skip_gguf_value(cursor: &mut Cursor<&[u8]>, value_type: &GgufType) -> Result<(), LociError> {
        match value_type {
            GgufType::Uint8 | GgufType::Int8 | GgufType::Bool => {
                cursor.seek(SeekFrom::Current(1))
                    .map_err(|e| LociError::IoWithContext { context: "skipping value", source: e })?;
            }
            GgufType::Uint16 | GgufType::Int16 => {
                cursor.seek(SeekFrom::Current(2))
                    .map_err(|e| LociError::IoWithContext { context: "skipping value", source: e })?;
            }
            GgufType::Uint32 | GgufType::Int32 | GgufType::Float32 => {
                cursor.seek(SeekFrom::Current(4))
                    .map_err(|e| LociError::IoWithContext { context: "skipping value", source: e })?;
            }
            GgufType::Uint64 | GgufType::Int64 | GgufType::Float64 => {
                cursor.seek(SeekFrom::Current(8))
                    .map_err(|e| LociError::IoWithContext { context: "skipping value", source: e })?;
            }
            GgufType::String => {
                let len = cursor.read_u64::<LittleEndian>()
                    .map_err(|e| LociError::IoWithContext { context: "reading string length for skip", source: e })?;
                cursor.seek(SeekFrom::Current(len as i64))
                    .map_err(|e| LociError::IoWithContext { context: "skipping string", source: e })?;
            }
            GgufType::Array => {
                let value_type = Self::read_gguf_type(cursor)?;
                let entries_num = cursor.read_u64::<LittleEndian>()
                    .map_err(|e| LociError::IoWithContext { context: "reading array length for skip", source: e })?;
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
        let current_pos = cursor.position() as i64;
        (current_pos + (alignment - 1)) & !(alignment - 1)
    }
}
