use std::collections::HashSet;

use async_fn_stream::try_fn_stream;
use brush_render::{gaussian_splats::inverse_sigmoid, sh::rgb_to_sh};
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use glam::{Vec3, Vec4};
use ply_rs::{
    parser::Parser,
    ply::{DefaultElement, ElementDef, Encoding, Header, Property, PropertyAccess},
};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncRead, BufReader};
use tokio_stream::{Stream, StreamExt};
use tokio_with_wasm::alias as tokio_wasm;
use tracing::trace_span;

use anyhow::{Context, Result};
use brush_render::gaussian_splats::Splats;

use crate::parsed_gaussian::ParsedGaussian;

pub struct ParseMetadata {
    pub up_axis: Option<Vec3>,
    pub total_splats: u32,
    pub frame_count: u32,
    pub current_frame: u32,
}

pub struct SplatMessage<B: Backend> {
    pub meta: ParseMetadata,
    pub splats: Splats<B>,
}

enum PlyFormat {
    Ply,
    Brush4DCompressed,
    SuperSplatCompressed,
}

fn interleave_coeffs(sh_dc: Vec3, sh_rest: &[f32], result: &mut Vec<f32>) {
    let channels = 3;
    let coeffs_per_channel = sh_rest.len() / channels;
    result.extend([sh_dc.x, sh_dc.y, sh_dc.z]);
    for i in 0..coeffs_per_channel {
        for j in 0..channels {
            let index = j * coeffs_per_channel + i;
            result.push(sh_rest[index]);
        }
    }
}

async fn parse_elem<T: AsyncBufRead + Unpin + 'static, E: PropertyAccess>(
    reader: &mut T,
    parser: &Parser<E>,
    encoding: Encoding,
    element: &ElementDef,
) -> tokio::io::Result<E> {
    match encoding {
        Encoding::Ascii => {
            let mut ascii_line = String::new();
            reader.read_line(&mut ascii_line).await?;
            let elem = parser.read_ascii_element(&ascii_line, element)?;
            ascii_line.clear();
            Ok(elem)
        }
        Encoding::BinaryBigEndian => parser.read_big_endian_element(reader, element).await,
        Encoding::BinaryLittleEndian => parser.read_little_endian_element(reader, element).await,
    }
}

struct TimeYield {
    last_yield: web_time::Instant,
    tick: usize,
}

impl TimeYield {
    fn new() -> Self {
        Self {
            last_yield: web_time::Instant::now(),
            tick: 0,
        }
    }

    /// Check if we need to yield. Should be called in loops.
    async fn try_yield(&mut self) {
        self.tick += 1;

        // Only check every so many iterations as checking the time isn't super cheap either.
        if self.tick % 1000 == 0 {
            let duration = web_time::Instant::now().duration_since(self.last_yield);
            // Try to yield every 5 milliseconds to keep things responsive.
            if duration.as_secs_f32() > 5e-3 {
                tokio_wasm::task::yield_now().await;
            }
        }
    }
}

pub fn load_splat_from_ply<T: AsyncRead + Unpin + 'static, B: Backend>(
    reader: T,
    subsample_points: Option<u32>,
    device: B::Device,
) -> impl Stream<Item = Result<SplatMessage<B>>> + 'static {
    // set up a reader, in this case a file.
    let mut reader = BufReader::new(reader);

    let _span = trace_span!("Read splats").entered();

    try_fn_stream(|emitter| async move {
        let header = Parser::<DefaultElement>::new()
            .read_header(&mut reader)
            .await?;

        // Parse some metadata.
        let up_axis = header
            .comments
            .iter()
            .filter_map(|c| match c.to_lowercase().strip_prefix("vertical axis: ") {
                Some("x") => Some(Vec3::X),
                Some("y") => Some(Vec3::NEG_Y),
                Some("z") => Some(Vec3::Z),
                _ => None,
            })
            .next_back();

        // Check whether there is a vertex header that has at least XYZ.
        let has_vertex = header.elements.iter().any(|el| el.name == "vertex");

        let ply_type = if has_vertex && header.elements.first().is_some_and(|el| el.name == "chunk")
        {
            PlyFormat::SuperSplatCompressed
        } else if has_vertex && header.elements.iter().any(|el| el.name == "delta_vertex_") {
            PlyFormat::Brush4DCompressed
        } else if has_vertex {
            PlyFormat::Ply
        } else {
            anyhow::bail!("Couldn't decide format of Ply file. Unknown Ply format.")
        };

        match ply_type {
            PlyFormat::Ply => {
                let mut stream =
                    std::pin::pin!(parse_ply(reader, subsample_points, device, header, up_axis));
                while let Some(splat) = stream.next().await {
                    emitter.emit(splat?).await;
                }
            }
            PlyFormat::Brush4DCompressed => {
                let mut stream = std::pin::pin!(parse_delta_ply(
                    reader,
                    subsample_points,
                    device,
                    header,
                    up_axis
                ));
                while let Some(splat) = stream.next().await {
                    emitter.emit(splat?).await;
                }
            }
            PlyFormat::SuperSplatCompressed => {
                let mut stream = std::pin::pin!(parse_compressed_ply(
                    reader,
                    subsample_points,
                    device,
                    header,
                    up_axis
                ));
                while let Some(splat) = stream.next().await {
                    emitter.emit(splat?).await;
                }
            }
        };

        Ok(())
    })
}

fn parse_ply<T: AsyncBufRead + Unpin + 'static, B: Backend>(
    mut reader: T,
    subsample_points: Option<u32>,
    device: B::Device,
    header: Header,
    up_axis: Option<Vec3>,
) -> impl Stream<Item = Result<SplatMessage<B>>> + 'static {
    try_fn_stream(|emitter| async move {
        let vertex = header.elements.first().context("No elements in header")?;
        if vertex.name != "vertex" {
            anyhow::bail!("First element must be 'vertex'")
        }

        let parser = Parser::<ParsedGaussian<false>>::new();

        let properties: HashSet<_> = vertex.properties.iter().map(|x| x.name.clone()).collect();
        let mut means = Vec::with_capacity(vertex.count);
        let mut log_scales = properties
            .contains("scale_0")
            .then(|| Vec::with_capacity(vertex.count));
        let mut rotations = properties
            .contains("rot_0")
            .then(|| Vec::with_capacity(vertex.count));
        let mut sh_coeffs = (properties.contains("f_dc_0") || properties.contains("red"))
            .then(|| Vec::with_capacity(vertex.count * 24));
        let mut opacity = properties
            .contains("opacity")
            .then(|| Vec::with_capacity(vertex.count));

        let update_every = vertex.count.div_ceil(20);

        let mut last_update = 0;

        let mut yielder = TimeYield::new();

        for i in 0..vertex.count {
            yielder.try_yield().await;

            // Doing this after first reading and parsing the points is quite wasteful, but
            // we do need to advance the reader.
            if let Some(subsample) = subsample_points {
                if i % subsample as usize != 0 {
                    continue;
                }
            }

            let splat = parse_elem(&mut reader, &parser, header.encoding, vertex).await?;

            if !splat.is_finite() {
                continue;
            }

            means.extend([splat.mean.x, splat.mean.y, splat.mean.z]);
            if let Some(scales) = &mut log_scales {
                scales.extend([splat.log_scale.x, splat.log_scale.y, splat.log_scale.z]);
            }
            if let Some(rotation) = &mut rotations {
                rotation.extend([
                    splat.rotation.w,
                    splat.rotation.x,
                    splat.rotation.y,
                    splat.rotation.z,
                ]);
            }
            if let Some(opacity) = &mut opacity {
                opacity.push(splat.opacity);
            }
            if let Some(sh_coeffs) = &mut sh_coeffs {
                interleave_coeffs(splat.sh_dc, &splat.sh_coeffs_rest, sh_coeffs);
            }

            if (i - last_update) >= update_every || i == vertex.count - 1 {
                let splats = Splats::from_raw(
                    means.clone(),
                    rotations.clone(),
                    log_scales.clone(),
                    sh_coeffs.clone(),
                    opacity.clone(),
                    &device,
                );
                emitter
                    .emit(SplatMessage {
                        meta: ParseMetadata {
                            total_splats: vertex.count as u32,
                            up_axis,
                            frame_count: 0,
                            current_frame: 0,
                        },
                        splats,
                    })
                    .await;

                last_update = i;
            }
        }

        Ok(())
    })
}

fn parse_compressed_ply<T: AsyncBufRead + Unpin + 'static, B: Backend>(
    mut reader: T,
    subsample_points: Option<u32>,
    device: B::Device,
    header: Header,
    up_axis: Option<Vec3>,
) -> impl Stream<Item = Result<SplatMessage<B>>> + 'static {
    #[derive(Default)]
    struct MinMax {
        min: Vec3,
        max: Vec3,
    }

    impl MinMax {
        fn dequant(&self, raw: Vec3) -> Vec3 {
            self.min + raw * (self.max - self.min)
        }
    }

    #[derive(Default)]
    struct QuantMeta {
        mean: MinMax,
        scale: MinMax,
        color: MinMax,
    }

    impl PropertyAccess for QuantMeta {
        fn new() -> Self {
            Self::default()
        }

        fn set_property(&mut self, key: &str, property: Property) {
            let ascii = key.as_bytes();
            let Property::Float(val) = property else {
                return;
            };
            match ascii {
                b"min_x" => self.mean.min.x = val,
                b"min_y" => self.mean.min.y = val,
                b"min_z" => self.mean.min.z = val,

                b"max_x" => self.mean.max.x = val,
                b"max_y" => self.mean.max.y = val,
                b"max_z" => self.mean.max.z = val,

                b"min_scale_x" => self.scale.min.x = val,
                b"min_scale_y" => self.scale.min.y = val,
                b"min_scale_z" => self.scale.min.z = val,

                b"max_scale_x" => self.scale.max.x = val,
                b"max_scale_y" => self.scale.max.y = val,
                b"max_scale_z" => self.scale.max.z = val,

                b"min_r" => self.color.min.x = val,
                b"min_g" => self.color.min.y = val,
                b"min_b" => self.color.min.z = val,

                b"max_r" => self.color.max.x = val,
                b"max_g" => self.color.max.y = val,
                b"max_b" => self.color.max.z = val,
                _ => {}
            }
        }
    }

    try_fn_stream(|emitter| async move {
        let quant_elem = header
            .elements
            .first()
            .context("Not enough elements in header")?;
        if quant_elem.name != "chunk" {
            anyhow::bail!("First element should be chunk compression metadata!");
        }

        let mut yielder = TimeYield::new();

        let parser = Parser::<QuantMeta>::new();
        let mut quant_metas = Vec::with_capacity(quant_elem.count);
        for _ in 0..quant_elem.count {
            yielder.try_yield().await;
            let quant_meta = parse_elem(&mut reader, &parser, header.encoding, quant_elem).await?;
            quant_metas.push(quant_meta);
        }

        let vertex = header
            .elements
            .get(1)
            .context("Not enough elements in header")?;
        if vertex.name != "vertex" {
            anyhow::bail!("Second element should be vertex compression metadata!");
        }

        let mut color_splat = None;
        let mut valid = vec![true; vertex.count];
        let mut sh_coeffs = Vec::with_capacity(vertex.count * 3);

        for i in 0..vertex.count {
            // Occasionally yield.
            yielder.try_yield().await;

            let update_every = vertex.count.div_ceil(20);
            let mut last_update = 0;

            for i in 0..vertex.count {
                // Occasionally yield.
                try_yield(i).await;

                // Doing this after first reading and parsing the points is quite wasteful, but
                // we do need to advance the reader.
                if let Some(subsample) = subsample_points {
                    if i % subsample as usize != 0 {
                        continue;
                    }
                }

                let quant_data = quant_metas
                    .get(i / 256)
                    .context("not enough quantization data to parse ply")?;

                let splat = parse_elem(&mut reader, &parser, header.encoding, vertex).await?;

                // Don't add invalid splats.
                if !splat.is_finite() {
                    valid[i] = false;
                    continue;
                }

                let mean = quant_data.mean.dequant(splat.mean);
                means.extend([mean.x, mean.y, mean.z]);

                let scale = quant_data.scale.dequant(splat.log_scale);
                log_scales.extend([scale.x, scale.y, scale.z]);
                rotations.extend([
                    splat.rotation.w,
                    splat.rotation.x,
                    splat.rotation.y,
                    splat.rotation.z,
                ]);

                // Compressed ply specifies things in post-activated values. Convert to pre-activated values.
                opacity.push(inverse_sigmoid(splat.opacity));

                // These come in as RGB colors. Convert to base SH coeffecients.
                let sh_dc = rgb_to_sh(quant_data.color.dequant(splat.sh_dc));
                sh_coeffs.extend([sh_dc.x, sh_dc.y, sh_dc.z]);

                // Occasionally send some updated splats.
                if (i - last_update) >= update_every || i == vertex.count - 1 {
                    let splats = Splats::from_raw(
                        means.clone(),
                        Some(rotations.clone()),
                        Some(log_scales.clone()),
                        Some(sh_coeffs.clone()),
                        Some(opacity.clone()),
                        &device,
                    );

                    color_splat = Some(splats.clone());

                    emitter
                        .emit(SplatMessage {
                            meta: ParseMetadata {
                                total_splats: vertex.count as u32,
                                up_axis,
                                frame_count: 0,
                                current_frame: 0,
                            },
                            splats,
                        })
                        .await;
                    last_update = i;
                }
            }
        }

        let Some(splats) = color_splat else {
            anyhow::bail!("No splat available in compressed ply file.");
        };

        if let Some(sh_vals) = header.elements.get(2) {
            // Bit of a hack - use the unquantized parser as that handles SH values. Really we don't need
            // the entire splat parser though.
            let parser = Parser::<ParsedGaussian<false>>::new();

            if sh_vals.name != "sh" {
                anyhow::bail!("Second element should be SH compression metadata!");
            }

            let mut splat_index = 0;

            let mut total_coeffs = Vec::with_capacity(sh_vals.count * 45);
            for i in 0..sh_vals.count {
                yielder.try_yield().await;

                if !valid[i] {
                    continue;
                }

                // Parse a splat - though nb only SH values will be used.
                let mut splat = parse_elem(&mut reader, &parser, header.encoding, sh_vals).await?;
                for coeff in &mut splat.sh_coeffs_rest {
                    *coeff = 8.0 * (*coeff - 0.5);
                }

                let dc = glam::vec3(
                    sh_coeffs[splat_index * 3],
                    sh_coeffs[splat_index * 3 + 1],
                    sh_coeffs[splat_index * 3 + 2],
                );
                interleave_coeffs(dc, &splat.sh_coeffs_rest, &mut total_coeffs);
                splat_index += 1;
            }

            let num_splats = splats.num_splats() as usize;
            let n_coeffs = total_coeffs.len() / num_splats;

            let mut splats = splats;
            // Apply new SH coeffs.
            splats.sh_coeffs = splats.sh_coeffs.map(|_| {
                Tensor::from_data(
                    TensorData::new(total_coeffs, [num_splats, n_coeffs / 3, 3]),
                    &device,
                )
            });

            // Send updates splats.
            emitter
                .emit(SplatMessage {
                    meta: ParseMetadata {
                        total_splats: vertex.count as u32,
                        up_axis,
                        frame_count: 0,
                        current_frame: 0,
                    },
                    splats,
                })
                .await;
        }

        Ok(())
    })
}

fn parse_delta_ply<T: AsyncBufRead + Unpin + 'static, B: Backend>(
    mut reader: T,
    subsample_points: Option<u32>,
    device: B::Device,
    header: Header,
    up_axis: Option<Vec3>,
) -> impl Stream<Item = Result<SplatMessage<B>>> + 'static {
    try_fn_stream(|emitter| async move {
        let parser = Parser::<ParsedGaussian<false>>::new();
        let mut yielder = TimeYield::new();

        // Check for frame count.
        let frame_count = header
            .elements
            .iter()
            .filter(|e| e.name.starts_with("delta_vertex_"))
            .count() as u32;

        let mut final_splat = None;
        let mut frame = 0;

        #[derive(Debug)]
        struct QuantMeta {
            mean: Vec3,
            rotation: Vec4,
            scale: Vec3,
        }

        let mut meta_min = QuantMeta {
            mean: Vec3::ZERO,
            rotation: Vec4::ZERO,
            scale: Vec3::ZERO,
        };
        let mut meta_max = QuantMeta {
            mean: Vec3::ONE,
            rotation: Vec4::ONE,
            scale: Vec3::ONE,
        };

        for element in &header.elements {
            let properties: HashSet<_> =
                element.properties.iter().map(|x| x.name.clone()).collect();

            let mut means = Vec::with_capacity(element.count);
            let mut log_scales = properties
                .contains("scale_0")
                .then(|| Vec::with_capacity(element.count * 3));
            let mut rotations = properties
                .contains("rot_0")
                .then(|| Vec::with_capacity(element.count * 4));
            let mut sh_coeffs = (properties.contains("f_dc_0") || properties.contains("red"))
                .then(|| Vec::with_capacity(element.count * 16));
            let mut opacity = properties
                .contains("opacity")
                .then(|| Vec::with_capacity(element.count));

            if element.name == "vertex" {
                let update_every = element.count.div_ceil(20);

                for i in 0..element.count {
                    yielder.try_yield().await;

                    // Occasionally send some updated splats.
                    if i % update_every == update_every - 1 {
                        emitter
                            .emit(SplatMessage {
                                meta: ParseMetadata {
                                    total_splats: element.count as u32,
                                    up_axis,
                                    frame_count,
                                    current_frame: frame,
                                },
                                splats: Splats::from_raw(
                                    means.clone(),
                                    rotations.clone(),
                                    log_scales.clone(),
                                    sh_coeffs.clone(),
                                    opacity.clone(),
                                    &device,
                                ),
                            })
                            .await;
                    }

                    // Doing this after first reading and parsing the points is quite wasteful, but
                    // we do need to advance the reader.
                    if let Some(subsample) = subsample_points {
                        if i % subsample as usize != 0 {
                            continue;
                        }
                    }

                    let splat = parse_elem(&mut reader, &parser, header.encoding, element).await?;

                    means.extend([splat.mean.x, splat.mean.y, splat.mean.z]);
                    if let Some(scales) = &mut log_scales {
                        scales.extend([splat.log_scale.x, splat.log_scale.y, splat.log_scale.z]);
                    }
                    if let Some(rotation) = &mut rotations {
                        rotation.extend([
                            splat.rotation.w,
                            splat.rotation.x,
                            splat.rotation.y,
                            splat.rotation.z,
                        ]);
                    }
                    if let Some(opacity) = &mut opacity {
                        opacity.push(splat.opacity);
                    }
                    if let Some(sh_coeffs) = &mut sh_coeffs {
                        interleave_coeffs(splat.sh_dc, &splat.sh_coeffs_rest, sh_coeffs);
                    }
                }
                let splats = Splats::from_raw(
                    means.clone(),
                    rotations.clone(),
                    log_scales.clone(),
                    sh_coeffs.clone(),
                    opacity.clone(),
                    &device,
                );
                final_splat = Some(splats.clone());
                emitter
                    .emit(SplatMessage {
                        meta: ParseMetadata {
                            total_splats: element.count as u32,
                            up_axis,
                            frame_count,
                            current_frame: frame,
                        },
                        splats,
                    })
                    .await;
            } else if element.name.starts_with("meta_delta_min_") {
                let splat = parse_elem(&mut reader, &parser, header.encoding, element).await?;
                meta_min.mean = splat.mean;
                meta_min.rotation = splat.rotation.into();
                meta_min.scale = splat.log_scale;
            } else if element.name.starts_with("meta_delta_max_") {
                let splat = parse_elem(&mut reader, &parser, header.encoding, element).await?;
                meta_max.mean = splat.mean;
                meta_max.rotation = splat.rotation.into();
                meta_max.scale = splat.log_scale;
            } else if element.name.starts_with("delta_vertex_") {
                let Some(splats) = final_splat.clone() else {
                    anyhow::bail!("Need to read base splat first.");
                };

                for _ in 0..element.count {
                    yielder.try_yield().await;

                    // The splat we decode is normed to 0-1 (if quantized), so rescale to
                    // actual values afterwards.
                    let splat_enc =
                        parse_elem(&mut reader, &parser, header.encoding, element).await?;

                    // Let's only animate transforms for now.
                    let mean = splat_enc.mean * (meta_max.mean - meta_min.mean) + meta_min.mean;
                    means.extend([mean.x, mean.y, mean.z]);

                    if let Some(rotation) = rotations.as_mut() {
                        let val: Vec4 = splat_enc.rotation.into();
                        let val = val * (meta_max.rotation - meta_min.rotation) + meta_min.rotation;
                        rotation.extend([val.w, val.x, val.y, val.z]);
                    }

                    if let Some(log_scales) = log_scales.as_mut() {
                        let scale = splat_enc.log_scale * (meta_max.scale - meta_min.scale)
                            + meta_min.scale;

                        log_scales.extend([scale.x, scale.y, scale.z]);
                    }
                    // Don't emit any intermediate states as it looks strange to have a torn state.
                }

                let n_splats = splats.num_splats() as usize;
                let means = Tensor::from_data(TensorData::new(means, [n_splats, 3]), &device)
                    + splats.means.val();

                // The encoding is just delta encoding in floats - nothing fancy
                // like actually considering the quaternion transform.
                let rotations = if let Some(rotations) = rotations {
                    Tensor::from_data(TensorData::new(rotations, [n_splats, 4]), &device)
                        + splats.rotation.val()
                } else {
                    splats.rotation.val()
                };

                let log_scales = if let Some(log_scales) = log_scales {
                    Tensor::from_data(TensorData::new(log_scales, [n_splats, 3]), &device)
                        + splats.log_scales.val()
                } else {
                    splats.log_scales.val()
                };

                // Emit newly animated splat.
                emitter
                    .emit(SplatMessage {
                        meta: ParseMetadata {
                            total_splats: element.count as u32,
                            up_axis,
                            frame_count,
                            current_frame: frame,
                        },
                        splats: Splats::from_tensor_data(
                            means,
                            rotations,
                            log_scales,
                            splats.sh_coeffs.val(),
                            splats.raw_opacity.val(),
                        ),
                    })
                    .await;

                frame += 1;
            }
        }

        Ok(())
    })
}
