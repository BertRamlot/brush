use super::DatasetZip;
use super::LoadDatasetArgs;
use crate::LoadInitArgs;
use crate::{clamp_img_to_max_size, DataStream, Dataset};
use anyhow::Context;
use anyhow::Result;
use async_fn_stream::try_fn_stream;
use brush_render::camera::{focal_to_fov, fov_to_focal, Camera};
use brush_render::gaussian_splats::Splats;
use brush_render::Backend;
use brush_train::scene::SceneView;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(serde::Deserialize)]
#[allow(unused)] // not reading camera distortions yet.
struct SyntheticScene {
    // Simple synthetic nerf camera model.
    camera_angle_x: Option<f64>,
    // Not really used atm.
    camera_model: Option<String>,

    // Nerfstudio doesn't mention this in their format? But fine to include really.
    ply_file_path: Option<String>,

    // Nerfstudio format
    //
    /// Focal length x
    fl_x: Option<f64>,
    /// Focal length y
    fl_y: Option<f64>,
    /// Principal point x
    cx: Option<f64>,
    /// Principal point y
    cy: Option<f64>,
    /// Image width
    w: Option<f64>,
    /// Image height
    h: Option<f64>,

    /// First radial distortion parameter used by [OPENCV, OPENCV_FISHEYE]
    k1: Option<f64>,
    /// Second radial distortion parameter used by [OPENCV, OPENCV_FISHEYE]
    k2: Option<f64>,
    /// Third radial distortion parameter used by [OPENCV_FISHEYE]
    k3: Option<f64>,
    /// Fourth radial distortion parameter used by [OPENCV_FISHEYE]
    k4: Option<f64>,
    /// First tangential distortion parameter used by [OPENCV]
    p1: Option<f64>,
    /// Second tangential distortion parameter used by [OPENCV]
    p2: Option<f64>,

    frames: Vec<FrameData>,
}

#[derive(serde::Deserialize)]
#[allow(unused)] // not reading camera distortions yet.
struct FrameData {
    // Nerfstudio format
    //
    /// Focal length x
    fl_x: Option<f64>,
    /// Focal length y
    fl_y: Option<f64>,
    /// Principal point x
    cx: Option<f64>,
    /// Principal point y
    cy: Option<f64>,
    /// Image width. Should be an integer but read as float, fine to truncate.
    w: Option<f64>,
    /// Image height. Should be an integer but read as float, fine to truncate.
    h: Option<f64>,

    // TODO: These are unused currently.
    /// First radial distortion parameter used by [OPENCV, OPENCV_FISHEYE]
    k1: Option<f64>,
    /// Second radial distortion parameter used by [OPENCV, OPENCV_FISHEYE]
    k2: Option<f64>,
    /// Third radial distortion parameter used by [OPENCV_FISHEYE]
    k3: Option<f64>,
    /// Fourth radial distortion parameter used by [OPENCV_FISHEYE]
    k4: Option<f64>,
    /// First tangential distortion parameter used by [OPENCV]
    p1: Option<f64>,
    /// Second tangential distortion parameter used by [OPENCV]
    p2: Option<f64>,

    transform_matrix: Vec<Vec<f32>>,
    file_path: String,
}

fn read_transforms_file(
    scene: SyntheticScene,
    transforms_path: PathBuf,
    archive: DatasetZip,
    load_args: &LoadDatasetArgs,
) -> Result<Vec<impl Future<Output = anyhow::Result<SceneView>>>> {
    let iter = scene
        .frames
        .into_iter()
        .take(load_args.max_frames.unwrap_or(usize::MAX))
        .map(move |frame| {
            let mut archive = archive.clone();
            let load_args = load_args.clone();
            let transforms_path = transforms_path.clone();

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

                // Read the imageat the specified path, fallback to default .png extension.
                let mut path = transforms_path
                    .clone()
                    .parent()
                    .unwrap()
                    .join(&frame.file_path);
                if path.extension().is_none() {
                    path = path.with_extension("png");
                }
                let img_buffer = archive.read_bytes_at_path(&path)?;

                let comp_span = tracing::trace_span!("Decompress image").entered();
                drop(comp_span);

                // Create a cursor from the buffer
                let mut image = tracing::trace_span!("Decode image")
                    .in_scope(|| image::load_from_memory(&img_buffer))?;

                let w = frame.w.or(scene.w).unwrap_or(image.width() as f64) as u32;
                let h = frame.h.or(scene.h).unwrap_or(image.height() as f64) as u32;

                if let Some(max_resolution) = load_args.max_resolution {
                    image = clamp_img_to_max_size(image, max_resolution);
                }

                let focal_x = frame
                    .fl_x
                    .or(scene.fl_x)
                    .or(scene.camera_angle_x)
                    .context("Must have a focal length of some kind.")?;

                // Read fov y or derive it from the input.
                let focal_y = frame
                    .fl_y
                    .or(scene.fl_y)
                    .unwrap_or(fov_to_focal(focal_x, w));

                let fovx = focal_to_fov(focal_x, w);
                let fovy = focal_to_fov(focal_y, h);

                let cx = frame.cx.or(scene.cx).unwrap_or(w as f64 / 2.0);
                let cy = frame.cy.or(scene.cy).unwrap_or(h as f64 / 2.0);

                let cuv = glam::vec2((cx / w as f64) as f32, (cy / h as f64) as f32);

                let view = SceneView {
                    name: frame.file_path.to_owned(),
                    camera: Camera::new(translation, rotation, fovx, fovy, cuv),
                    image: Arc::new(image),
                };
                anyhow::Result::<SceneView>::Ok(view)
            }
        });

    Ok(iter.collect())
}

pub fn read_dataset<B: Backend>(
    mut archive: DatasetZip,
    init_args: &LoadInitArgs,
    load_args: &LoadDatasetArgs,
    device: &B::Device,
) -> Result<(DataStream<Splats<B>>, DataStream<Dataset>)> {
    log::info!("Loading nerf synthetic dataset");

    let transforms_path = archive.find_with_extension(".json", "_train")?;
    let train_scene = serde_json::from_reader(archive.file_at_path(&transforms_path)?)?;

    let load_args = load_args.clone();

    let dataset_stream = try_fn_stream(|emitter| async move {
        let mut train_views = vec![];
        let mut eval_views = vec![];

        let load_args = load_args.clone();
        let train_handles =
            read_transforms_file(train_scene, transforms_path, archive.clone(), &load_args)?;

        let transforms_path = archive.find_with_extension(".json", "_train")?;
        let val_scene = serde_json::from_reader(archive.file_at_path(&transforms_path)?)?;
        let val_stream = read_transforms_file(val_scene, transforms_path, archive, &load_args).ok();

        log::info!("Loading transforms_test.json");
        // Not entirely sure yet if we want to report stats on both test
        // and eval, atm this skips "transforms_test.json" even if it's there.

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

    let device = device.clone();
    let splat_stream = try_fn_stream(|_| async move {
        // Not implemented atm.
        let _ = init_args;
        let _ = device;
        Ok(())
    });

    Ok((Box::pin(splat_stream), Box::pin(dataset_stream)))
}
