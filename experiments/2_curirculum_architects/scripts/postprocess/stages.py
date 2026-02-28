import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def build_stage_manifests(
    shuffled_index: pa.Table,
    curriculum: dict,
    out_prefix: str,
    filesystem,
):
    stages = curriculum["growth_schedule"]["stages"]
    profiles = curriculum["growth_schedule"]["stage_profiles"]

    for stage in stages:
        name = stage["name"]
        profile_name = stage["profile"]
        weights = profiles[profile_name]["band_weights"]

        print(f"Building stage {name}")

        parts = []

        for band, ratio in weights.items():
            band_rows = shuffled_index.filter(pc.equal(shuffled_index["band"], band))

            take = int(len(band_rows) * ratio)
            parts.append(band_rows.slice(0, take))

        stage_table = pa.concat_tables(parts)

        out = f"{out_prefix.rstrip('/')}/stage_{name}.parquet"
        with filesystem.open(out, "wb") as f:
            pq.write_table(stage_table, f)

        print(f"Wrote {out} ({len(stage_table)} rows)")
