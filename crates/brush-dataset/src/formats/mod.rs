use crate::{
    splat_import::load_splat_from_ply, zip::DatasetZip, Dataset, LoadDatasetArgs, LoadInitArgs,
};
use anyhow::Result;
use brush_render::{gaussian_splats::Splats, Backend};
use std::{io::Cursor, path::Path, pin::Pin};
use tokio_stream::Stream;

pub mod colmap;
pub mod nerf_synthetic;

// A dynamic stream of datasets
type DataStream<T> = Pin<Box<dyn Stream<Item = Result<T>> + Send + 'static>>;

pub fn load_dataset<B: Backend>(
    mut archive: DatasetZip,
    init_args: &LoadInitArgs,
    load_args: &LoadDatasetArgs,
    device: &B::Device,
) -> anyhow::Result<(DataStream<Splats<B>>, DataStream<Dataset>)> {
    let streams = nerf_synthetic::read_dataset(archive.clone(), init_args, load_args, device)
        .or_else(|e| {
            log::info!("Not a NeRF synthetic dataset ({e}), trying to load as Colmap.");
            colmap::load_dataset::<B>(archive.clone(), init_args, load_args, device)
        });

    let Ok(streams) = streams else {
        anyhow::bail!("Couldn't parse dataset as any format. Only some formats are supported.")
    };

    // If there's an init.ply definitey override the init stream with that. Nb:
    // this ignores the specified number of SH channels atm.
    let init_ply = archive.read_bytes_at_path(Path::new("init.ply"));

    let init_stream = if let Ok(data) = init_ply {
        let splat_stream = load_splat_from_ply(Cursor::new(data), device.clone());
        Box::pin(splat_stream)
    } else {
        streams.0
    };

    Ok((init_stream, streams.1))
}
