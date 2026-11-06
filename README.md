# Chongqing Watershed MCP

Source code and data access information for rainfall-runoff and regional
slope-stability modelling in Chongqing Municipality, China. The software runs
on Windows and the web interface defaults to English.

| Directory | Purpose |
| --- | --- |
| [config/](config/) | Agent, spatial and hydrology settings, parameter presets and quality checks |
| [data/](data/) | Sample ERA5 rainfall, a reference river layer and links to source datasets |
| [docs/](docs/) | Installation, usage and technical guides |
| [scripts/](scripts/) | Service launchers, setup checks and a runnable synthetic basin example |
| [src/](src/) | Agent, spatial and hydrology services, and their shared input format |
| [tests/](tests/) | Tests for the agent, services, data exchange and security |
| [workers/](workers/) | Background workers for GIS processing, river extraction and hydrology |

## Get the code and data

Use the repository's **Code** menu to clone it or download a ZIP. The
[data guide](data/README.md) explains where to obtain ERA5, CLCD and a public
30 m elevation model, and lists the included files and restricted inputs.
See the [quick start](docs/getting-started.md) for installation and a synthetic
example. Third-party code notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
