"""Runs with GRASS's own Python; raster IDs preserve holes and multipart basins."""
from pathlib import Path
import sys
import grass.script as gs


def main():
    dem, output, threshold = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    output.mkdir(parents=True, exist_ok=True)
    gs.run_command("r.in.gdal", input=str(dem), output="dem", overwrite=True, quiet=True)
    gs.run_command("g.region", raster="dem", flags="a", quiet=True)
    gs.run_command("r.watershed", flags="s", elevation="dem", threshold=max(1, threshold),
                   basin="basin", drainage="drainage", accumulation="accumulation",
                   stream="stream", overwrite=True, quiet=True)
    for name in ("basin", "drainage", "accumulation", "stream"):
        gs.run_command("r.out.gdal", input=name, output=str(output / (name + ".tif")),
                       format="GTiff", type="Float64" if name == "accumulation" else "Int32",
                       nodata=0, createopt="COMPRESS=LZW", overwrite=True, quiet=True)
    (output / "version.txt").write_text(str(gs.version()), encoding="utf-8")


if __name__ == "__main__":
    main()
