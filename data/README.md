# Data access

Version 1.0 supports study areas within Chongqing Municipality, China.
This directory provides an ERA5 precipitation sample, a reference river layer
and download links for ERA5 and CLCD.
Original providers retain their respective terms and attribution requirements.

| Dataset | Access | Material supplied here |
| --- | --- | --- |
| ERA5 hourly data on single levels | [Copernicus Climate Data Store, DOI 10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47) | A 24-hour excerpt of the project's ERA5-derived precipitation table for 4 August 2009 |
| China Land Cover Dataset (CLCD), 2023 layer, 30 m | [Yang and Huang, Zenodo version 1.0.3, DOI 10.5281/zenodo.12779975](https://doi.org/10.5281/zenodo.12779975) | Original download and citation record. Select the 2023 file from the record and follow its attribution terms. |
| Reference river layer for this release | [Natural Earth 1:10m Rivers + Lake Centerlines v5.0.0](https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-rivers-lake-centerlines/) | `chongqing_reference_rivers.shp` and its sidecars, in the public domain |

## ERA5 example

`water/逐时降水_宽表/ERA5_重庆逐时降水_2009.csv` contains the header
and 24 rows (00:00 through 23:00 in the project's local-time convention) from
the existing project table. The file uses a UTF-8 BOM. The first column is
time. The other column names are grid coordinates in longitude_latitude
order, with hourly precipitation in millimetres. Use the CDS catalogue entry
for the original `total_precipitation` variable, reference and licence.
The excerpt preserves the project values and does not cover the historical
search period used to select the storm.

Contains modified Copernicus Climate Change Service information derived from
ERA5. Changes in this repository consist of selecting the date from the
project's derived table. For the original product, cite Hersbach et al.,
*ERA5 hourly data on single levels from 1940 to present*, Copernicus Climate
Change Service (C3S) Climate Data Store, DOI 10.24381/cds.adbb2d47.

## CLCD for a local analysis

Download the 2023 raster from the authors' record. The spatial service expects
a land-use polygon layer with a supported class field. Use the class legend
from the source product, prepare the layer for your authorized area, and set
the resulting file name under `datasets.landuse` in
`config/spatial.default.yaml`. Retain the shapefile sidecars (`.dbf`,
`.shx`, `.prj` and `.cpg` if supplied). The local study's processed
land-use layer is outside the distribution.

## Reference river layer

The bundled Natural Earth layer contains six generalized line features in a
rectangle covering Chongqing and adjacent terrain (105–111°E, 28–33°N).
It was clipped to the rectangle in WGS84 and retains the source `name`
field. This is a 1:10 million scale map layer for visual orientation.
The rectangular clip used no study boundaries or restricted watercourse data.
The source ZIP has SHA-256
`ded71b01870855ccfe19b51f2ec14c9bb48fae23c0e9f3c11974d426433b5c38`.

When using an external `CQ_DATA_ROOT`, copy the reference shapefile and all
four sidecars there to display it. The reference layer is configured under
`datasets.display_rivers`. Supply an authorized, detailed named watercourse
layer under `datasets.rivers` for river-name search and associated operations.
HydroRIVERS is available separately from [the original provider](https://www.hydrosheds.org/products/hydrorivers)
under its own license, which restricts standalone redistribution.

## Local inputs and reproduction limits

The authors lack permission to redistribute the following study materials.
Access requires separate authorization from the respective holders.

| Material | Role and access |
| --- | --- |
| Chongqing 12.5 m DEM | The study elevation raster was provided by the Chongqing Institute of Geology and Mineral Resources. Contact the provider for access. A public alternative is described below. |
| May 2026 hourly rain-gauge series | The ten gauges supplied rainfall forcing for the May 2026 simulation. Request the original observations from their data holder. |
| Mapped landslide polygons | The 82 polygons were used for spatial comparison and ROC analysis. Request the inventory from its data holder. |
| Administrative and study boundaries, and the study's named watercourse layer | These files supported area selection and spatial processing. Supply authorized local layers for your own analysis. |
| Study input packages and derived outputs | The retained packages and spatial products incorporate restricted inputs. Their public redistribution is outside the scope of this release. |

Reproducing the reported Yongchuan results requires the original study inputs.
The synthetic basin demonstrates the numerical workflow on generated inputs.

## Substitute a public 30 m elevation model

For a new analysis within Chongqing, users can independently obtain the
[Copernicus DEM GLO-30 Public product](https://doi.org/10.5270/ESA-c5d3d65).
This is a surface elevation model at approximately 30 m posting. Select tiles
covering the analysis area and surrounding terrain needed for drainage checks.
Mosaic them if needed and save a GeoTIFF with CRS and NoData metadata, using a
metre-based projected CRS suitable for the analysis. Keep the effective model
grid at 30 m or coarser and record the product version and processing steps.
Place the GeoTIFF in the local data directory and set `datasets.dem` in
`config/spatial.default.yaml` to its relative file name. Set
`swmm.delineation_resolution_m` and `stability.calculation_resolution_m`
in `config/hydro.default.yaml` to 30 m or coarser. Retain the provider's
attribution when using the data.

Recheck drainage boundaries, channel locations, and the effects of changing
terrain resolution before interpreting results. The replacement yields a new
analysis whose values may differ from the 12.5 m study case. The DEM is
downloaded separately from the provider.

For a new analysis within Chongqing with authorized data, edit the dataset names in
`config/spatial.default.yaml` and set `CQ_DATA_ROOT` in the PowerShell
session used to launch the software.

| Input role | Format |
| --- | --- |
| Elevation | GeoTIFF with an explicit CRS |
| Land use and river layers | Shapefiles with sidecars and documented class or name fields |
| Regional boundary | Shapefile with a CRS |
| Administrative boundaries | WGS84 GeoJSON |
| Hourly rainfall | Year-labelled wide CSV files, with timestamps and coordinate columns following the ERA5 excerpt |

Keep authorized working data outside the repository.
