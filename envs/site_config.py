import os.path
import sys
import json
from dataclasses import dataclass
from pathlib import Path

'''
Loads lab- or installation-specific config from the site_config.json file
When adding a new field, include it with a type hint in SiteConfig below, as well as in the json file.
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

def _load_config(fname) -> SiteConfig:
    this_package = Path(__file__).resolve().parent
    data = json.loads(Path(os.path.join(this_package, fname)).read_text())
    v = sys.version_info
    version_key = 'version_specific'
    config_dict = {}
    for key, value in data.items():
        if type(value) == dict and version_key in value:
            config_dict[key] = value[version_key][str(v.minor)]
        else:
            config_dict[key] = value
    return SiteConfig(**config_dict)

#edit the file name here to swithc between config files
CONFIG = _load_config("site_config.json")
if __name__ == "__main__":
    print(CONFIG)