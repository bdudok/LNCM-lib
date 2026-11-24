import os.path
import sys
import json
from dataclasses import dataclass
from pathlib import Path

'''
Loads lab- or installation-specific config from the site_config.json file
For code completion, when adding a new field, include it with a type hint in SiteConfig below, as well as in the json.
If there are settings that depend on the python version, set its value in the json like this:
    {'version_specific': {'6': 'sample_rate', '11': 'sample_frequency'}}
'''

@dataclass(frozen=True)
class SiteConfig:
    EDF_fs_key: str #for reading the sample rate from an edf file. Used only when opening an EDF file.
    python36_executable: str #for calling SIMA functions.
    alt_processed_paths: list # LoadRegistered uses this to find archived motion-corrected movies in backups
    alt_raw_paths: list # scripts use this to find archived raw movies in backups
    ledcal: float #100 % led power in mW, for calibrating photostimulation intensity
    scratch_disk: str #saving temporary files
    nwb_scratch: str #temporary processed folders of NWBImagingSession (on a data drive, not the boot drive)
    lab_identifier: str #name of the lab, for generating dataset identifiers
    uv_env_dir: str #envs/update_env.bat builds each user's lncm env in <uv_env_dir>/<username>/lncm
    job_queue: dict #job queue manager: api_port (localhost), dashboard_port (lab network), data_dir (manager's database)
    use_baserow: bool = False #False: code that can run without Baserow skips it silently (job queue Task updates, animal info of AnalysisResult)

def _load_config(fname) -> SiteConfig:
    this_package = Path(__file__).resolve().parent
    data = json.loads(Path(os.path.join(this_package, fname)).read_text())
    v = sys.version_info
    version_key = 'version_specific'
    config_dict = {}
    for key, value in data.items():
        if type(value) is dict and version_key in value:
            config_dict[key] = value[version_key][str(v.minor)]
        else:
            config_dict[key] = value
    return SiteConfig(**config_dict)

#edit the file name here to swithc between config files
CONFIG = _load_config("site_config.json")
if __name__ == "__main__":
    print(CONFIG)