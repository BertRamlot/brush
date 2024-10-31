use burn_wgpu::{AutoGraphicsApi, RuntimeOptions};
use eframe::egui_wgpu::{WgpuConfiguration, WgpuSetup};

pub async fn create_wgpu_setup() -> burn_wgpu::WgpuSetup {
    burn_wgpu::init_setup_async::<AutoGraphicsApi>(
        &burn_wgpu::WgpuDevice::DefaultDevice,
        RuntimeOptions {
            tasks_max: 64,
            memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
        },
    )
    .await
}

pub async fn create_wgpu_egui_config() -> WgpuConfiguration {
    let setup = create_wgpu_setup().await;

    WgpuConfiguration {
        wgpu_setup: WgpuSetup::Existing {
            instance: setup.instance,
            adapter: setup.adapter,
            device: setup.device,
            queue: setup.queue,
        },
        ..Default::default()
    }
}
