use half::f16;

use crate::gguf_types::GgufTensorInfo;
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use candle_core;
use candle_core::{Tensor, Device, DType};

pub struct LlmModel;

impl LlmModel {
    pub fn load_tensor(&self, mmap: &Mmap, tensor_info: &GgufTensorInfo, tensor_offset_start: i64) -> anyhow::Result<Tensor> {
        let offset = (tensor_offset_start + tensor_info.offset) as usize;
        let raw_bytes = &mmap[(offset as usize)..(offset as usize + 20)];
            println!("Raw bytes at offset {}: {:?}", offset, raw_bytes);
        // let mut cursor = Cursor::new(mmap);
        // cursor.seek(SeekFrom::Start(offset as u64))?;

        let ggml_type = tensor_info.ggml_type;
        let shapes = tensor_info.shapes.iter()
            .map(|&v| v as usize)
            .collect::<Vec<usize>>();
        let entries_count: usize = shapes.iter()
            .product();

        let tensor = match ggml_type {
            0 => { // F32
                // let mut buffer = vec![0f32; entries_count];
                // for entry in &mut buffer {
                //     *entry = cursor.read_f32::<LittleEndian>()?;
                // }
                // Tensor::from_vec(buffer, shapes, &Device::cuda_if_available(0)?)?
                let byte_len = entries_count * 4;
                let data = &mmap[offset..offset + byte_len];
                Tensor::from_raw_buffer(data, DType::F32, &shapes, &Device::cuda_if_available(0)?)?
                
            },
            1 => { // F16
                // let mut buffer = vec![0f32; entries_count];
                // for entry in &mut buffer {
                //     let bits = cursor.read_u16::<LittleEndian>()?;
                //     let entry_f16 = f16::from_bits(bits);
                //     *entry = entry_f16.to_f32();
                // }
                // Tensor::from_vec(buffer, shapes, &Device::cuda_if_available(0)?)?
                
                let byte_len = entries_count * 2;
                let data = &mmap[offset..offset + byte_len];
                Tensor::from_raw_buffer(data, DType::F16, &shapes, &Device::cuda_if_available(0)?)?
                
            },
            _ => anyhow::bail!("Unsupported tensor dtype: {}", ggml_type),
        };

        Ok(tensor)
    }
}