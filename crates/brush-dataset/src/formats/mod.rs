use crate::{splat_import::load_splat_from_ply, zip::DatasetZip, Dataset, LoadDatasetArgs};
use anyhow::Result;
use brush_render::{gaussian_splats::Splats, Backend};
use std::{io::Cursor, path::Path, pin::Pin};
use tokio_stream::Stream;

pub mod colmap;
pub mod nerfstudio;

// A dynamic stream of datasets
type DataStream<T> = Pin<Box<dyn Stream<Item = Result<T>> + Send + 'static>>;

pub fn load_dataset<B: Backend>(
    mut archive: DatasetZip,
    load_args: &LoadDatasetArgs,
    device: &B::Device,
) -> anyhow::Result<(DataStream<Splats<B>>, DataStream<Dataset>)> {
    let streams = nerfstudio::read_dataset(archive.clone(), load_args, device)
        .or_else(|_| colmap::load_dataset::<B>(archive.clone(), load_args, device));

    let Ok(streams) = streams else {
        anyhow::bail!("Couldn't parse dataset as any format. Only some formats are supported.")
    };

    // If there's an init.ply definitey override the init stream with that.
    let init_ply = archive.read_bytes_at_path(Path::new("init.ply"));

    let init_stream = if let Ok(data) = init_ply {
        let splat_stream = load_splat_from_ply(Cursor::new(data), device.clone());
        Box::pin(splat_stream)
    } else {
        streams.0
    };

    Ok((init_stream, streams.1))
}
