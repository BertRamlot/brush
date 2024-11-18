use std::collections::HashSet;

use async_fn_stream::try_fn_stream;
use brush_render::Backend;
use glam::{Quat, Vec3};
use ply_rs::{
    parser::Parser,
    ply::{Property, PropertyAccess},
};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio_stream::Stream;
use tracing::trace_span;

use anyhow::Result;
use brush_render::gaussian_splats::Splats;

pub(crate) struct GaussianData {
    pub(crate) means: Vec3,
    pub(crate) scale: Vec3,
    pub(crate) opacity: f32,
    pub(crate) rotation: Quat,
    pub(crate) sh_dc: [f32; 3],
    // NB: This is in the inria format, aka [channels, coeffs]
    // not [coeffs, channels].
    pub(crate) sh_coeffs_rest: Vec<f32>,
}

impl PropertyAccess for GaussianData {
    fn new() -> Self {
        GaussianData {
            means: Vec3::ZERO,
            scale: Vec3::ZERO,
            opacity: 0.0,
            rotation: Quat::IDENTITY,
            sh_dc: [0.0, 0.0, 0.0],
            sh_coeffs_rest: Vec::new(),
        }
    }

    fn set_property(&mut self, key: &str, property: Property) {
        let value = if let Property::Float(v) = property {
            v
        } else if let Property::UChar(v) = property {
            let scale = match key {
                "red" => 255.0,
                "green" => 255.0,
                "blue" => 255.0,
                _ => {
                    log::error!("Invalid ply property. {key} as a uchar is not supported.");
                    return;
                }
            };
            v as f32 / scale
        } else {
            return;
        };

        match key {
            "x" => self.means[0] = value,
            "y" => self.means[1] = value,
            "z" => self.means[2] = value,
            "scale_0" => self.scale[0] = value,
            "scale_1" => self.scale[1] = value,
            "scale_2" => self.scale[2] = value,
            "opacity" => self.opacity = value,
            "rot_0" => self.rotation.w = value,
            "rot_1" => self.rotation.x = value,
            "rot_2" => self.rotation.y = value,
            "rot_3" => self.rotation.z = value,
            "f_dc_0" | "red" => self.sh_dc[0] = value,
            "f_dc_1" | "green" => self.sh_dc[1] = value,
            "f_dc_2" | "blue" => self.sh_dc[2] = value,
            _ if key.starts_with("f_rest_") => {
                if let Ok(idx) = key["f_rest_".len()..].parse::<u32>() {
                    if idx >= self.sh_coeffs_rest.len() as u32 {
                        self.sh_coeffs_rest.resize(idx as usize + 1, 0.0);
                    }
                    self.sh_coeffs_rest[idx as usize] = value;
                }
            }
            _ => (),
        }
    }

    fn get_float(&self, key: &str) -> Option<f32> {
        match key {
            "x" => Some(self.means[0]),
            "y" => Some(self.means[1]),
            "z" => Some(self.means[2]),
            "scale_0" => Some(self.scale[0]),
            "scale_1" => Some(self.scale[1]),
            "scale_2" => Some(self.scale[2]),
            "opacity" => Some(self.opacity),
            "rot_0" => Some(self.rotation.w),
            "rot_1" => Some(self.rotation.x),
            "rot_2" => Some(self.rotation.y),
            "rot_3" => Some(self.rotation.z),
            "f_dc_0" => Some(self.sh_dc[0]),
            "f_dc_1" => Some(self.sh_dc[1]),
            "f_dc_2" => Some(self.sh_dc[2]),
            _ if key.starts_with("f_rest_") => {
                if let Ok(idx) = key["f_rest_".len()..].parse::<usize>() {
                    self.sh_coeffs_rest.get(idx).copied()
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

fn interleave_coeffs(sh_dc: [f32; 3], sh_rest: &[f32]) -> Vec<f32> {
    let channels = 3;
    let coeffs_per_channel = sh_rest.len() / channels;
    let mut result = Vec::with_capacity(sh_rest.len() + 3);
    result.extend(sh_dc);

    for i in 0..coeffs_per_channel {
        for j in 0..channels {
            let index = j * coeffs_per_channel + i;
            result.push(sh_rest[index]);
        }
    }
    result
}

pub fn load_splat_from_ply<T: AsyncRead + Unpin + 'static, B: Backend>(
    reader: T,
    device: B::Device,
) -> impl Stream<Item = Result<Splats<B>>> + 'static {
    // set up a reader, in this case a file.
    let mut reader = BufReader::new(reader);

    let update_every = 10000;
    let _span = trace_span!("Read splats").entered();
    let gaussian_parser = Parser::<GaussianData>::new();

    try_fn_stream(|emitter| async move {
        let header = gaussian_parser.read_header(&mut reader).await?;

        for element in &header.elements {
            if element.name == "vertex" {
                let properties: HashSet<_> =
                    element.properties.iter().map(|x| x.name.clone()).collect();

                if ["x", "y", "z"].into_iter().any(|p| !properties.contains(p)) {
                    Err(anyhow::anyhow!("Invalid splat ply. Missing properties!"))?
                }

                let n_sh_coeffs = (3 + element
                    .properties
                    .iter()
                    .filter_map(|x| {
                        x.name
                            .strip_prefix("f_rest_")
                            .and_then(|x| x.parse::<u32>().ok())
                    })
                    .max()
                    .unwrap_or(0)) as usize;

                let mut means = Vec::with_capacity(element.count);
                let mut scales = properties
                    .contains("scale_0")
                    .then(|| Vec::with_capacity(element.count));
                let mut rotation = properties
                    .contains("rot_0")
                    .then(|| Vec::with_capacity(element.count));
                let mut sh_coeffs = (properties.contains("f_dc_0") || properties.contains("red"))
                    .then(|| Vec::with_capacity(element.count * n_sh_coeffs));
                let mut opacity = properties
                    .contains("opacity")
                    .then(|| Vec::with_capacity(element.count));

                for i in 0..element.count {
                    let splat = match header.encoding {
                        ply_rs::ply::Encoding::Ascii => {
                            let mut line = String::new();
                            reader.read_line(&mut line).await?;
                            gaussian_parser.read_ascii_element(&line, element)?
                        }
                        ply_rs::ply::Encoding::BinaryBigEndian => {
                            gaussian_parser
                                .read_big_endian_element(&mut reader, element)
                                .await?
                        }
                        ply_rs::ply::Encoding::BinaryLittleEndian => {
                            gaussian_parser
                                .read_little_endian_element(&mut reader, element)
                                .await?
                        }
                    };

                    let sh_coeffs_interleaved =
                        interleave_coeffs(splat.sh_dc, &splat.sh_coeffs_rest);

                    means.push(splat.means);
                    if let Some(scales) = scales.as_mut() {
                        scales.push(splat.scale);
                    }
                    if let Some(rotation) = rotation.as_mut() {
                        rotation.push(splat.rotation);
                    }
                    if let Some(opacity) = opacity.as_mut() {
                        opacity.push(splat.opacity);
                    }
                    if let Some(sh_coeffs) = sh_coeffs.as_mut() {
                        sh_coeffs.extend(sh_coeffs_interleaved);
                    }

                    // Occasionally send some updated splats.
                    if i % update_every == update_every - 1 {
                        let splats = Splats::from_raw(
                            means.clone(),
                            rotation.clone(),
                            scales.clone(),
                            sh_coeffs.clone(),
                            opacity.clone(),
                            &device,
                        );

                        emitter.emit(splats).await;
                    }

                    // Ocassionally yield.
                    if i % 250 == 0 {
                        tokio::task::yield_now().await;
                    }
                }

                let splats = Splats::from_raw(means, rotation, scales, sh_coeffs, opacity, &device);

                if splats.num_splats() == 0 {
                    Err(anyhow::anyhow!("No splats found"))?;
                }

                emitter.emit(splats).await;
            }
        }

        Ok(())
    })
}
