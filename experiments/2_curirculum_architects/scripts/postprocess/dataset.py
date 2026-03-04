import pyarrow.parquet as pq
import s3fs
from torch.utils.data import Dataset


class CurriculumDataset(Dataset):
    def __init__(self, manifest_path):
        self.fs = s3fs.S3FileSystem()
        self.manifest = pq.read_table(manifest_path, filesystem=self.fs).to_pandas()

        # cache parquet readers
        self.files = {}

    def __len__(self):
        return len(self.manifest)

    def _get_file(self, path):
        if path not in self.files:
            self.files[path] = pq.ParquetFile(path, filesystem=self.fs)
        return self.files[path]

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        pf = self._get_file(row.file)

        # Read exactly ONE row
        table = pf.read_row_group(row.row // pf.metadata.row_group(0).num_rows)

        return table.slice(row.row % table.num_rows, 1).to_pylist()[0]
