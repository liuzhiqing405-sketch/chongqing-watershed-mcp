# Getting started

## Installation

Use Windows x64, Python 3.12 and EPA SWMM 5.2.4. Install GRASS GIS and make its launcher available on PATH.
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
formats in [the data guide](../data/README.md). Set `CQ_DATA_ROOT` to that
directory. To start the two MCP services and web application:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_agent_latest.ps1 -ConfigureInWeb
```

Open <http://127.0.0.1:8780>. For live language-model interaction, enter your
own API endpoint, model identifier and key in the model settings. Supported
formats include OpenAI Chat Completions, OpenAI Responses, Anthropic Messages
and Gemini GenerateContent. The key is held in process memory. The web interface
and new sessions default to English. Geographic names and data fields may retain
their source spelling. The numerical example runs independently of the web interface.
