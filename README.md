# Chongqing Watershed MCP

A Windows research implementation of a natural-language workflow for regional
slope-stability screening. An agent coordinates spatial preparation and
rainfall-runoff and factor-of-safety calculations through two independent
Model Context Protocol (MCP) services. Numerical calculations run in the
engineering services.

Version **1.0** currently supports study areas within **Chongqing Municipality,
China**. Read the [data guide](data/README.md) for restricted study inputs,
the reference river layer and the option to use a public 30 m elevation model.

| Path | Contents |
| --- | --- |
| `src/slope_evaluation_agent/` | Agent, prompts, task state, result verification and web interface |
| `src/chongqing_spatial_mcp/` | Study boundaries, spatial preparation and rainfall packaging |
| `src/watershed_hydro_mcp/` | Delineation, SWMM, factor of safety, sensitivity analysis and quality checks |
| `src/watershed_exchange/` | Model input package format and validation |
| `config/`, `scripts/`, `workers/` | Configuration templates, launchers and worker processes |
| `tests/` | Tests and generated test fixtures |
| [data/README.md](data/README.md) | ERA5, CLCD, the public reference river layer and the limits on reproducing the published case |

## Installation

Use Windows x64 and Python 3.12. The manuscript reports Python 3.12.7 and
EPA SWMM 5.2.4. Install GRASS GIS and make its launcher available on PATH.
The project searches for `grass`, `grass85`, `grass84` and `grass83`.
If needed, set `executables.grass_cli` in `config/hydro.default.yaml`
to the installed launcher.

From PowerShell at the repository root, with Python 3.12 on PATH:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e ".[web,llm,test]"
```

Install [EPA SWMM 5.2.4 for Windows x64](https://www.epa.gov/water-research/storm-water-management-model-swmm#software).
Place `runswmm.exe`, `swmm5.dll` and the matching `vcomp140.dll`
from your installation together under `native/swmm/`. The files are local
dependencies. To use an external installation with this layout, set
`WHS_NATIVE_ROOT` to the parent of its `swmm` directory.

## Run a synthetic basin

The smoke example generates a small valley, land cover, boundary and rainfall
from code, then runs delineation, SWMM and factor-of-safety calculation using
the normal hydrology service job manager. It requires GRASS and SWMM and runs
without a language-model account. It writes results and stage logs below the
requested output directory, which must be empty or new.

```powershell
& .\.venv\Scripts\python.exe scripts/run_synthetic_case.py --output runs/synthetic
```

On completion, the command reports the job status, the minimum factor of
safety and the modeled area with an event-minimum factor of safety below 1.
The GeoTIFF is kept with the run outputs. These results describe a generated
test case. The verification run completed with a minimum factor of safety of 2.000000 and 0.00 m² below 1.

## Use the web interface with your own inputs

Prepare an authorized local data directory using the dataset roles and input
formats in [the data guide](data/README.md). Set `CQ_DATA_ROOT` to that
directory. To start the two MCP services and web application:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_agent_latest.ps1 -ConfigureInWeb
```

Open <http://127.0.0.1:8780>. For live language-model interaction, enter your
own API endpoint, model identifier and key in the model settings. Supported formats include OpenAI Chat Completions, OpenAI Responses, Anthropic Messages and Gemini GenerateContent. The key is
held in process memory. The web interface and new sessions default to English. Geographic names and data fields can retain their source spelling. The numerical example
above can be run independently of the web interface.

## Release status

This is the local submission candidate for version 1.0. The public release,
source code license and manuscript-specific archive identifier are pending.
Record the code and configuration that produced each reported experiment
when preparing the manuscript archive.
Third-party code notices are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
