use super::DatasetZip;
use super::LoadDatasetArgs;
use crate::{clamp_img_to_max_size, DataStream, Dataset};
use anyhow::Context;
use anyhow::Result;
use async_fn_stream::try_fn_stream;
use brush_render::camera::{focal_to_fov, fov_to_focal, Camera};
use brush_train::scene::SceneView;
use std::future::Future;
use std::io::Read;
use std::sync::Arc;

#[derive(serde::Deserialize)]
struct SyntheticScene {
    camera_angle_x: Option<f64>,
    frames: Vec<FrameData>,
}

#[derive(serde::Deserialize)]
struct FrameData {
    fl_x: Option<f64>,
    fl_y: Option<f64>,

    cx: Option<f64>,
    cy: Option<f64>,

    transform_matrix: Vec<Vec<f32>>,
    file_path: String,
}

fn read_transforms_file(
    mut archive: DatasetZip,
    name: &'static str,
    load_args: &LoadDatasetArgs,
) -> Result<Vec<impl Future<Output = anyhow::Result<SceneView>>>> {
    let base_path = archive
        .find_base_path(name)
        .context("No transforms file found")?;

    let transform_path = base_path.join(name);

    let transform_buf = {
        let mut transforms_file = archive.file_at_path(&transform_path)?;
        let mut transform_buf = String::new();
        transforms_file.read_to_string(&mut transform_buf)?;
        transform_buf
    };

    let scene_train: SyntheticScene = serde_json::from_str(&transform_buf)?;
    let fovx = scene_train.camera_angle_x;

    let iter = scene_train
        .frames
        .into_iter()
        .take(load_args.max_frames.unwrap_or(usize::MAX))
        .map(move |frame| {
            let base_path = base_path.clone();
            let mut archive = archive.clone();
            let load_args = load_args.clone();

            async move {
                // NeRF 'transform_matrix' is a camera-to-world transform
                let transform_matrix: Vec<f32> =
                    frame.transform_matrix.iter().flatten().copied().collect();
                let mut transform = glam::Mat4::from_cols_slice(&transform_matrix).transpose();

                // Swap basis to go from z-up, left handed (a la OpenCV) to our kernel format
                // (right-handed, y-down).
                transform.y_axis *= -1.0;
                transform.z_axis *= -1.0;

                transform = glam::Mat4::from_rotation_x(std::f32::consts::PI / 2.0) * transform;

                let (_, rotation, translation) = transform.to_scale_rotation_translation();

                let image_path = &base_path.join(frame.file_path.to_owned() + ".png");

                let comp_span = tracing::trace_span!("Decompress image").entered();
                let img_buffer = archive.read_bytes_at_path(image_path)?;
                drop(comp_span);

                // Create a cursor from the buffer
                let mut image = tracing::trace_span!("Decode image")
                    .in_scope(|| image::load_from_memory(&img_buffer))?;

                if let Some(max_resolution) = load_args.max_resolution {
                    image = clamp_img_to_max_size(image, max_resolution);
                }

                let fovy = focal_to_fov(fov_to_focal(fovx, image.width()), image.height());

                let view = SceneView {
                    name: image_path.to_str().context("Invalid filename")?.to_owned(),
                    camera: Camera::new(translation, rotation, fovx, fovy, glam::vec2(0.5, 0.5)),
                    image: Arc::new(image),
                };
                anyhow::Result::<SceneView>::Ok(view)
            }
        });

    Ok(iter.collect())
}

pub fn read_dataset(
    archive: DatasetZip,
    load_args: &LoadDatasetArgs,
) -> Result<DataStream<Dataset>> {
    log::info!("Loading nerf synthetic dataset");

    let load_args = load_args.clone();
    let train_handles = read_transforms_file(archive.clone(), "transforms_train.json", &load_args)?;

    let stream = try_fn_stream(|emitter| async move {
        let mut train_views = vec![];
        let mut eval_views = vec![];

        log::info!("Loading transforms_test.json");
        // Not entirely sure yet if we want to report stats on both test
        // and eval, atm this skips "transforms_test.json" even if it's there.
        let val_stream = read_transforms_file(archive, "transforms_val.json", &load_args).ok();

        for (i, handle) in train_handles.into_iter().enumerate() {
            log::info!("Getting train img");

            if let Some(eval_period) = load_args.eval_split_every {
                // Include extra eval images only when the dataset doesn't have them.
                if i % eval_period == 0 && val_stream.is_some() {
                    eval_views.push(handle.await?);
                } else {
                    train_views.push(handle.await?);
                }
            } else {
                train_views.push(handle.await?);
            }

            emitter
                .emit(Dataset::from_views(train_views.clone(), eval_views.clone()))
                .await;
        }

        if let Some(val_stream) = val_stream {
            for handle in val_stream {
                eval_views.push(handle.await?);
                emitter
                    .emit(Dataset::from_views(train_views.clone(), eval_views.clone()))
                    .await;
            }
        }

        Ok(())
    });

    Ok(Box::pin(stream))
}
