'''Choice of the spike tag of the exported LFP tables (NWB_converter.ExportImagingSession.lfp_event_tag)'''
import os

from NWB_converter.ExportImagingSession import LATEST_TAG, lfp_event_tag

PREFIX = 'BL6-1_2026-01-01_LFP_33'


def make(folder, name, mtime):
    fn = folder / name
    fn.write_text('')
    os.utime(fn, (mtime, mtime))


def test_tag_preference_and_latest(tmp_path):
    make(tmp_path, f'{PREFIX}_Ch1_spiketimes.xlsx', 100)
    make(tmp_path, f'{PREFIX}_AE_1_Ch1_seizure_times.xlsx', 300)
    make(tmp_path, f'{PREFIX}_AE_1_Ch1_spiketimes.xlsx', 200)
    make(tmp_path, f'{PREFIX}_AE_2_Ch2_seizure_times_curated.xlsx', 50)
    # another session whose prefix starts with this one
    make(tmp_path, f'{PREFIX}7_Ch1_spiketimes.xlsx', 900)
    assert lfp_event_tag(tmp_path, PREFIX, 1, ['AE_2', LATEST_TAG]) == 'AE_1'
    assert lfp_event_tag(tmp_path, PREFIX, 2, ['AE_2', LATEST_TAG]) == 'AE_2'
    assert lfp_event_tag(tmp_path, PREFIX, 1, [None]) is None
    assert lfp_event_tag(tmp_path, PREFIX, 2, [None]) is False
    assert lfp_event_tag(tmp_path, PREFIX, 3, ['AE_2', LATEST_TAG]) is False
    make(tmp_path, f'{PREFIX}_Ch1_seizure_times.xlsx', 400)
    assert lfp_event_tag(tmp_path, PREFIX, 1, [LATEST_TAG]) is None
