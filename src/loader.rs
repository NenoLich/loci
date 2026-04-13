use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::path::Path;
use std::io::{Cursor, Read, Seek};
use crate::gguf_types::{GgufHeaders, GgufType, GgufValue, GgufKVMeta, GgufInfo, GgufTensorInfo};
use num_traits::FromPrimitive;
use std::io::SeekFrom;

pub struct Loader;

impl Loader {
    pub fn load_gguf_info(&self, path: impl AsRef<Path>, first_k_tensors: usize, verbose: bool) -> anyhow::Result<GgufInfo> {
        let file = File::open(&path)?;
        let mmap: Mmap = unsafe { MmapOptions::new().map(&file)? };
        let mut cursor = Cursor::new(&mmap[..]);

        // 1. Load Headers
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            anyhow::bail!("Invalid GGUF file: magic bytes mismatch");
        }

        let version = cursor.read_u32::<LittleEndian>()?;
        let tensor_count = cursor.read_u64::<LittleEndian>()?;
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()?;
        
        let gguf_headers = GgufHeaders {
            path: path.as_ref().to_string_lossy().into_owned(),
            magic: String::from_utf8_lossy(&magic).into_owned(),
            version, 
            tensor_count, 
            metadata_kv_count
        };

        // 2. Load Metadata
        let mut gguf_kv_meta_vec: Vec<GgufKVMeta> = vec![];

        for _ in 0..gguf_headers.metadata_kv_count {
            let key = self.read_gguf_string(&mut cursor)?;
            let value_type = self.read_gguf_type(&mut cursor)?;
            let value = self.get_gguf_value(&mut cursor, &value_type)?;

            gguf_kv_meta_vec.push(GgufKVMeta {key, value_type, value});
        }

        // 3. Load Tensor info

        let mut gguf_tensor_info: Vec<GgufTensorInfo> = vec![];

        for _ in 0..gguf_headers.tensor_count {
            let name = self.read_gguf_string(&mut cursor)?;
            let n_dims = self.read_gguf_int32(&mut cursor)?;
            let mut shapes: Vec<i64> = vec![];
            for _ in 0..n_dims {
                let shape = self.read_gguf_int64(&mut cursor)?;
                shapes.push(shape);
            }

            let ggml_type = self.read_gguf_int32(&mut cursor)?;
            let offset = self.read_gguf_int64(&mut cursor)?;
            gguf_tensor_info.push(GgufTensorInfo {name, n_dims, shapes, ggml_type, offset});
        }

        // 4. Get the tensor offset start
        let alignment = self.get_byte_alignment(&gguf_kv_meta_vec);
        let gguf_tensor_offset_start = self.get_pos_with_byte_alignment(&mut cursor, alignment);

        if verbose {
            self.print_gguf_headers(&gguf_headers);
            self.print_gguf_kv_meta(&gguf_kv_meta_vec);
            self.print_gguf_tensor_info(&gguf_tensor_info, first_k_tensors, gguf_headers.tensor_count)?;
        }

        let gguf_info = GgufInfo {
            headers: gguf_headers, 
            kv_meta: gguf_kv_meta_vec, 
            tensor_info: gguf_tensor_info, 
            tensor_offset_start: gguf_tensor_offset_start
        };
        
        anyhow::Ok(gguf_info)
    }

    fn print_gguf_headers(&self, headers: &GgufHeaders) {
        println!("🦀 loci Model Loader");
        println!("─────────────────────────────────");
        println!("File: {}", headers.path);
        println!("Magic: {}", headers.magic);
        println!("Version: {}", headers.version);
        println!("Tensor Count: {}", headers.tensor_count);
        println!("Metadata KV Count: {}", headers.metadata_kv_count);
        println!("─────────────────────────────────");
        println!("✅ Model file is valid GGUF format!");
    }

    fn print_gguf_kv_meta(&self, kv_meta: &[GgufKVMeta]) {
        println!("\n📋 Metadata:");
        println!("─────────────────────────────────");
        for entry in kv_meta {
            if !matches!(entry.value.as_slice(), Some(array) if array.len() >= 16) {
                println!("{}: {} = {}", entry.key, entry.value_type, entry.value);
            } else {
                println!("{}: {} = [MORE THEN 16 ENTRIES]", entry.key, entry.value_type);
            }  
        }
        println!("─────────────────────────────────");
    }

    fn print_gguf_tensor_info(&self, tensor_info: &[GgufTensorInfo], first_k: usize, tensor_count: u64) -> anyhow::Result<()> {
        let first_k_to_show = first_k.min(tensor_info.len());
        println!("📦 Tensors (first {} of {}):", first_k_to_show, tensor_count);
        println!("─────────────────────────────────");
        for i in 0..first_k_to_show {
            println!("[{}] {} | n_dims: {} | shape: {:?} | type: {} | offset: {:#x}",
                 i, 
                 tensor_info[i].name, 
                 tensor_info[i].n_dims, 
                 tensor_info[i].shapes, 
                 tensor_info[i].ggml_type, 
                 tensor_info[i].offset
                );     
        }
        println!("─────────────────────────────────");

        anyhow::Ok(())
    }

    fn read_gguf_uint8(&self, cursor: &mut Cursor<&[u8]>) -> anyhow::Result<u8> {
        anyhow::Ok(cursor.read_u8()?)
    }

    fn read_gguf_int8(&self, cursor: &mut Cursor<&[u8]>) -> anyhow::Result<i8> {
        anyhow::Ok(cursor.read_i8()?)
    }

    fn read_gguf_uint16(&self, cursor: &mut Cursor<&[u8]>) -> anyhow::Result<u16> {
        anyhow::Ok(cursor.read_u16::<LittleEndian>()?)
    }

    fn read_gguf_int16(&self, cursor: &mut Cursor<&[u8]>) -> anyhow::Result<i16> {
        anyhow::Ok(cursor.read_i16::<LittleEndian>()?)
    }

    fn read_gguf_uint32(&self, cursor: &mut Cursor<&[u8]>) -> anyhow::Result<u32> {
        anyhow::Ok(cursor.read_u32::<LittleEndian>()?)
    }

    fn read_gguf_int32(&self, cursor: &mut Cursor<&[u8]>) -> anyhow::Result<i32> {
        anyhow::Ok(cursor.read_i32::<LittleEndian>()?)
    }

    fn read_gguf_float32(&self, cursor: &mut Cursor<&[u8]>) -> anyhow::Result<f32> {
        anyhow::Ok(cursor.read_f32::<LittleEndian>()?)
    }

    fn read_gguf_uint64(&self, cursor: &mut Cursor<&[u8]>) -> anyhow::Result<u64> {
        anyhow::Ok(cursor.read_u64::<LittleEndian>()?)
    }

    fn read_gguf_int64(&self, cursor: &mut Cursor<&[u8]>) -> anyhow::Result<i64> {
        anyhow::Ok(cursor.read_i64::<LittleEndian>()?)
    }

    fn read_gguf_float64(&self, cursor: &mut Cursor<&[u8]>) -> anyhow::Result<f64> {
        anyhow::Ok(cursor.read_f64::<LittleEndian>()?)
    }

    fn read_gguf_bool(&self, cursor: &mut Cursor<&[u8]>) -> anyhow::Result<bool> {
        let value = cursor.read_i8()? != 0;
        anyhow::Ok(value)
    }

    fn read_gguf_array(&self, cursor: &mut Cursor<&[u8]>) -> anyhow::Result<Vec<GgufValue>> {
        let value_type = self.read_gguf_type(cursor)?;
        let entries_num = cursor.read_u64::<LittleEndian>()? as usize;
        let mut values: Vec<GgufValue> = vec![];
        for _ in 0..entries_num {
            let value = self.get_gguf_value(cursor, &value_type)?;
            values.push(value);
        }

        anyhow::Ok(values)
    }

    fn read_gguf_string(&self, cursor: &mut Cursor<&[u8]>) -> anyhow::Result<String> {
        let len = cursor.read_u64::<LittleEndian>()? as usize;
        let mut buffer = vec![0u8; len];
        cursor.read_exact(&mut buffer)?;
        anyhow::Ok(String::from_utf8(buffer)?)
    }

    fn read_gguf_type(&self, cursor: &mut Cursor<&[u8]>) -> anyhow::Result<GgufType> {
        let gguf_type_n = cursor.read_i32::<LittleEndian>()?;
        GgufType::from_i32(gguf_type_n)
            .ok_or_else(|| anyhow::anyhow!("Unknown GGUF type ID: {}", gguf_type_n))
    }

    fn get_gguf_value(&self, cursor: &mut Cursor<&[u8]>, value_type: &GgufType) -> anyhow::Result<GgufValue> {
        let value = match value_type {
                GgufType::Uint8 => GgufValue::Uint8(self.read_gguf_uint8(cursor)?),
                GgufType::Int8 => GgufValue::Int8(self.read_gguf_int8(cursor)?),
                GgufType::Uint16 => GgufValue::Uint16(self.read_gguf_uint16(cursor)?),
                GgufType::Int16 => GgufValue::Int16(self.read_gguf_int16(cursor)?),
                GgufType::Uint32 => GgufValue::Uint32(self.read_gguf_uint32(cursor)?),
                GgufType::Int32 => GgufValue::Int32(self.read_gguf_int32(cursor)?),
                GgufType::Float32 => GgufValue::Float32(self.read_gguf_float32(cursor)?),
                GgufType::Bool => GgufValue::Bool(self.read_gguf_bool(cursor)?),
                GgufType::String => GgufValue::String(self.read_gguf_string(cursor)?),
                GgufType::Array => GgufValue::Array(self.read_gguf_array(cursor)?),
                GgufType::Uint64 => GgufValue::Uint64(self.read_gguf_uint64(cursor)?),
                GgufType::Int64 => GgufValue::Int64(self.read_gguf_int64(cursor)?),
                GgufType::Float64 => GgufValue::Float64(self.read_gguf_float64(cursor)?),
            };

        anyhow::Ok(value)
    }

    fn skip_gguf_value(&self, cursor: &mut Cursor<&[u8]>, value_type: &GgufType) -> anyhow::Result<()> {
        match value_type {
            GgufType::Uint8 | GgufType::Int8 | GgufType::Bool => 
                { cursor.seek(SeekFrom::Current(1))?; }
            GgufType::Uint16 | GgufType::Int16 => 
                { cursor.seek(SeekFrom::Current(2))?; }
            GgufType::Uint32 | GgufType::Int32 | GgufType::Float32 => 
                { cursor.seek(SeekFrom::Current(4))?; }
            GgufType::Uint64 | GgufType::Int64 | GgufType::Float64 => 
                { cursor.seek(SeekFrom::Current(8))?; }
            GgufType::String => {
                let len = cursor.read_u64::<LittleEndian>()?;
                cursor.seek(SeekFrom::Current(len as i64))?;
            }
            GgufType::Array => {
                let value_type = self.read_gguf_type(cursor)?;
                let entries_num = cursor.read_u64::<LittleEndian>()?;
                for _ in 0..entries_num {
                    self.skip_gguf_value(cursor, &value_type)?;
                }
            }
        };

        anyhow::Ok(())
    }

    fn get_byte_alignment(&self, kv_meta: &[GgufKVMeta]) -> i64 {
        let alignment = kv_meta.iter()
            .find(|&entry| entry.key == "general.alignment")
            .and_then(|f| f.value.as_i64())
            .unwrap_or(32);
        alignment
    }

    fn get_pos_with_byte_alignment(&self, cursor: &mut Cursor<&[u8]>, alignment: i64) -> i64 {
        let current_pos = cursor.position() as i64;
        let aligned_pos = (current_pos + (alignment - 1)) & !(alignment - 1);
        aligned_pos
    }
}