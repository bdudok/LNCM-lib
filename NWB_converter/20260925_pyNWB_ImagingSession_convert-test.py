from pathlib import Path

from NWB_converter.NWBExporter import NWBExporter

# the template config is in the repo; each dataset's config can be saved in the dataset folder
prefix = 'Dag47_2026-04-22_goal_128'
config_file = Path(__file__).resolve().parent / 'export_config_template_2ch-lean.yaml'

exporter = NWBExporter(prefix, config_file)
exporter.run()
