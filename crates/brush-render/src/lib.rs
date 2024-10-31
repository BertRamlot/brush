#![allow(clippy::too_many_arguments)]
#![allow(clippy::single_range_in_vec_init)]
use brush_kernel::bitcast_tensor;
use burn::backend::Autodiff;
use burn::prelude::Tensor;
use burn::tensor::{ElementConversion, Int, Shape, TensorPrimitive};
use burn_jit::JitBackend;
use burn_wgpu::{JitTensor, WgpuRuntime};
use camera::Camera;

mod dim_check;
mod kernels;
mod safetensor_utils;
mod shaders;

pub mod bounding_box;
pub mod camera;
pub mod gaussian_splats;
pub mod render;

#[derive(Debug, Clone)]
pub struct RenderAux {
    pub uniforms_buffer: JitTensor<WgpuRuntime, u32>,
    pub projected_splats: JitTensor<WgpuRuntime, f32>,
    pub num_intersections: JitTensor<WgpuRuntime, u32>,
    pub num_visible: JitTensor<WgpuRuntime, u32>,
    pub final_index: JitTensor<WgpuRuntime, u32>,
    pub cum_tiles_hit: JitTensor<WgpuRuntime, u32>,
    pub tile_bins: JitTensor<WgpuRuntime, u32>,
    pub compact_gid_from_isect: JitTensor<WgpuRuntime, u32>,
    pub global_from_compact_gid: JitTensor<WgpuRuntime, u32>,
}

#[derive(Debug, Clone)]
pub struct RenderStats {
    pub num_visible: u32,
    pub num_intersections: u32,
}

impl RenderAux {
    pub async fn read_num_visible(&self) -> u32 {
        Tensor::<JitBackend<WgpuRuntime, f32, i32>, 1, Int>::from_primitive(bitcast_tensor(
            self.num_visible.clone(),
        ))
        .into_scalar_async()
        .await
        .elem()
    }

    pub async fn read_num_intersections(&self) -> u32 {
        Tensor::<JitBackend<WgpuRuntime, f32, i32>, 1, Int>::from_primitive(bitcast_tensor(
            self.num_intersections.clone(),
        ))
        .into_scalar_async()
        .await
        .elem()
    }

    pub fn read_tile_depth(&self) -> Tensor<JitBackend<WgpuRuntime, f32, i32>, 2, Int> {
        let bins = Tensor::from_primitive(bitcast_tensor(self.tile_bins.clone()));
        let [ty, tx, _] = bins.dims();
        let max = bins.clone().slice([0..ty, 0..tx, 1..2]).squeeze(2);
        let min = bins.clone().slice([0..ty, 0..tx, 0..1]).squeeze(2);
        max - min
    }
}

// Custom operations in Burn work by extending the backend with an extra func.
pub trait Backend: burn::tensor::backend::Backend {
    /// Render splats to a buffer.
    ///
    /// This projects the gaussians, sorts them, and rasterizes them to a buffer, in a\
    /// differentiable way.
    /// The arguments are all passed as raw tensors. See [`Splats`] for a convenient Module that wraps this fun
    /// The ['xy_dummy'] variable is only used to carry screenspace xy gradients.
    /// This function can optionally render a "u32" buffer, which is a packed RGBA (8 bits per channel)
    /// buffer. This is useful when the results need to be displayed immediatly.
    fn render_splats(
        cam: &Camera,
        img_size: glam::UVec2,
        means: Tensor<Self, 2>,
        xy_grad_dummy: Tensor<Self, 2>,
        xy_grad_norm_dummy: Tensor<Self, 1>,
        log_scales: Tensor<Self, 2>,
        quats: Tensor<Self, 2>,
        sh_coeffs: Tensor<Self, 3>,
        raw_opacity: Tensor<Self, 1>,
        background: glam::Vec3,
        render_u32_buffer: bool,
    ) -> (Tensor<Self, 3>, RenderAux);
}

pub trait AutodiffBackend: burn::tensor::backend::AutodiffBackend + Backend {}
impl<B: Backend> AutodiffBackend for Autodiff<B> where burn::backend::Autodiff<B>: Backend {}

pub type PrimaryBackend = JitBackend<WgpuRuntime, f32, i32>;
pub static VALIDATION_TASK_HANDLE: std::sync::OnceLock<
    async_std::channel::Sender<ValidationMessage>,
> = std::sync::OnceLock::new();

pub fn init_validation_task() {
    async fn assert_not_nan<const D: usize>(
        tensor: JitTensor<WgpuRuntime, f32>,
        shape: [usize; D],
        max: Option<JitTensor<WgpuRuntime, u32>>,
        description: String,
    ) {
        let tensor = Tensor::<PrimaryBackend, 1>::from_primitive(TensorPrimitive::Float(tensor));
        let tensor = tensor.reshape::<D, _>(Shape::new(shape));

        let tensor = if let Some(max) = max {
            let max_data = Tensor::<PrimaryBackend, D, Int>::from_primitive(bitcast_tensor(max));
            let index = max_data.into_scalar_async().await.elem::<u32>();

            if index == 0 {
                return;
            }

            tensor.slice([0..index as usize])
        } else {
            tensor
        };

        if tensor.contains_nan().into_scalar_async().await {
            log::error!("Tensor contains NaN: {}", description);
        }
    }

    let _ = VALIDATION_TASK_HANDLE.get_or_init(|| {
        let (sender, receiver) = async_std::channel::unbounded();

        let fut = async move {
            while let Ok(msg) = receiver.recv().await {
                match msg {
                    ValidationMessage::ValidateNotNan {
                        tensor,
                        shape,
                        max,
                        description,
                    } => {
                        if shape.len() == 1 {
                            assert_not_nan::<1>(
                                tensor,
                                shape.try_into().unwrap(),
                                max,
                                description,
                            )
                            .await;
                        } else if shape.len() == 2 {
                            assert_not_nan::<2>(
                                tensor,
                                shape.try_into().unwrap(),
                                max,
                                description,
                            )
                            .await;
                        } else if shape.len() == 3 {
                            assert_not_nan::<3>(
                                tensor,
                                shape.try_into().unwrap(),
                                max,
                                description,
                            )
                            .await;
                        } else if shape.len() == 4 {
                            assert_not_nan::<4>(
                                tensor,
                                shape.try_into().unwrap(),
                                max,
                                description,
                            )
                            .await;
                        }
                    }
                }
            }
        };
        #[cfg(target_arch = "wasm32")]
        async_std::task::spawn_local(fut);
        #[cfg(not(target_arch = "wasm32"))]
        async_std::task::spawn(fut);

        sender
    });
}

#[derive(Debug)]
pub enum ValidationMessage {
    ValidateNotNan {
        tensor: JitTensor<WgpuRuntime, f32>,
        shape: Vec<usize>,
        max: Option<JitTensor<WgpuRuntime, u32>>,
        description: String,
    },
}

pub fn validate_not_nan<const D: usize>(
    tensor: JitTensor<WgpuRuntime, f32>,
    shape: [usize; D],
    max: Option<JitTensor<WgpuRuntime, u32>>,
    description: &str,
) {
    init_validation_task();
    VALIDATION_TASK_HANDLE
        .get()
        .unwrap()
        .try_send(ValidationMessage::ValidateNotNan {
            tensor,
            shape: shape.to_vec(),
            max,
            description: description.to_owned(),
        })
        .unwrap();
}
