// Currently, we make all datasets go through a zip file [1]
// This class helps working with an archive as a somewhat more regular filesystem.
//
// [1] really we want to just read directories.
// The reason is that picking directories isn't supported on
// rfd on wasm, nor is drag-and-dropping folders in egui.
use std::{
    io::{Cursor, Read},
    path::{Path, PathBuf},
    sync::Arc,
};

use zip::{
    read::ZipFile,
    result::{ZipError, ZipResult},
    ZipArchive,
};

#[derive(Clone)]
pub struct ZipData {
    data: Arc<Vec<u8>>,
}

type ZipReader = Cursor<ZipData>;

impl AsRef<[u8]> for ZipData {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

impl ZipData {
    pub fn open_for_read(&self) -> ZipReader {
        Cursor::new(self.clone())
    }
}

impl From<Vec<u8>> for ZipData {
    fn from(value: Vec<u8>) -> Self {
        Self {
            data: Arc::new(value),
        }
    }
}

pub(crate) fn normalized_path(path: &Path) -> PathBuf {
    Path::new(path)
        .components()
        .skip_while(|c| matches!(c, std::path::Component::CurDir))
        .collect::<PathBuf>()
}

#[derive(Clone)]
pub struct DatasetZip {
    archive: ZipArchive<Cursor<ZipData>>,
}

// TODO: This is all awfully ad-hoc.
impl DatasetZip {
    pub fn from_data(data: Vec<u8>) -> ZipResult<Self> {
        let zip_data = ZipData::from(data);
        let archive = ZipArchive::new(zip_data.open_for_read())?;
        Ok(Self { archive })
    }

    pub(crate) fn file_names(&self) -> impl Iterator<Item = &str> + '_ {
        self.archive
            .file_names()
            // stupic macOS.
            .filter(|p| !p.contains("__MACOSX"))
    }

    pub(crate) fn find_with_extension(
        &self,
        extension: &str,
        contains: &str,
    ) -> anyhow::Result<PathBuf> {
        let names: Vec<_> = self
            .file_names()
            .filter(|name| name.ends_with(extension))
            .collect();

        if names.len() == 1 {
            return Ok(Path::new(names[0]).to_owned());
        }

        let names: Vec<_> = names
            .iter()
            .filter(|name| name.contains(contains))
            .collect();

        if names.len() == 1 {
            return Ok(Path::new(names[0]).to_owned());
        }

        anyhow::bail!("Failed to find file ending in {extension} maybe containing {contains}.");
    }

    pub(crate) fn file_at_path<'a>(&'a mut self, path: &Path) -> Result<ZipFile<'a>, ZipError> {
        let name = self
            .archive
            .file_names()
            .find(|name| path == Path::new(name))
            .ok_or(ZipError::FileNotFound)?;
        let name = name.to_owned();
        self.archive.by_name(&name)
    }

    pub(crate) fn read_bytes_at_path(&mut self, path: &Path) -> anyhow::Result<Vec<u8>> {
        let mut buffer = vec![];
        self.file_at_path(path)?.read_to_end(&mut buffer)?;
        Ok(buffer)
    }

    pub(crate) fn find_base_path(&self, search_path: &str) -> Option<PathBuf> {
        for file in self.archive.file_names() {
            let path = normalized_path(Path::new(file));
            if path.ends_with(search_path) {
                return path
                    .ancestors()
                    .nth(Path::new(search_path).components().count())
                    .map(|x| x.to_owned());
            }
        }
        None
    }
}
