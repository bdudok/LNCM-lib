import sys

'''
Define installation-specific code format here
'''

v = sys.version_info
assert v.major == 3
lfp_config = {}
if v.minor == 6:
    lfp_config['EDF_fs_key'] = 'sample_rate'
elif v.minor == 11:
    lfp_config['EDF_fs_key'] = 'sample_frequency'