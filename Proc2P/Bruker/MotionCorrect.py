import numpy
import os, shutil
import suite2p
# run using suite2p conda env but s2p package, not source

from Proc2P.Bruker import PreProc
from tifffile import imwrite

'''This function runs the pre processing and motion correction, should be called bys script in suite2p repo.
Should be called from suite2p env.
See: _TEMPLATE_SCRIPT/20241126_BatchRun_MotionCorrect_Template.py
'''


def run(ops, session_df, pre_only, overwrite_previous, overwrite_preproc, ref_ch):
    username = os.environ.get('USERNAME')
    print(f'Running suite2p version {suite2p.version} as {username}')
    fast_disk = f'E:/S2P/{username}/'
    ops['fast_disk'] = fast_disk

    for i, item in session_df.iterrows():
        dpath = item['Raw.Path']
        processed_path = item['Processed.Path']
        imID = item['Image.ID']
        prefix = imID[:-4]
        btag = imID[-3:]
        print(prefix)
        if not os.path.exists(processed_path):
            os.mkdir(processed_path)

        # preprocess session
        s = PreProc.PreProc(dpath, processed_path, prefix, btag, overwrite=overwrite_preproc)
        if pre_only:
            continue

        if os.path.exists(fast_disk):
            if 'E:' in fast_disk:
                shutil.rmtree(fast_disk)

        if overwrite_previous:
            ofn = os.path.join(s.procpath, 'suite2p')
            if os.path.exists(ofn):
                shutil.rmtree(ofn)
            for dirname in ('_trace_1-ch1', '_trace_1-ch0'):
                ofn = os.path.join(s.procpath, prefix + dirname)
                if os.path.exists(ofn):
                    shutil.rmtree(ofn)
            suffix_list = ('_trace_1.npy', '_saved_roi_1.npy', '_saved_roi_STICA-R.npy', '_saved_roi_STICA-G.npy',
                           '_avgmax.tif', '_preview.tif', '_registered_Ch2.bin', '_registered_Ch1.bin')
            for suffix in suffix_list:
                ofn = os.path.join(s.procpath, prefix + suffix)
                if os.path.exists(ofn):
                    os.remove(ofn)

        filelist = [fn for fn in os.listdir(s.dpath) if ((prefix in fn) and ('.ome.tif' in fn))]

        # session ops
        db = {
            'fs': s.si['framerate'],
            'data_path': [s.dpath, ],
            'tiff_list': filelist,
            'save_path0': s.procpath
        }

        dual_channel = len(s.channelnames) > 1
        if dual_channel:
            db['input_format'] = 'bruker'
            db['bruker'] = True
            ops['nchannels'] = 2
            ops['align_by_chan'] = s.channelnames.index(ref_ch)

        # run motion correction
        output_ops = suite2p.run_s2p(ops=ops, db=db)

        # clean up output
        mean1 = output_ops['meanImg'] / output_ops['meanImg'].max() * 255
        previewname = s.procpath + prefix + f'_preview.tif'
        preview = numpy.zeros((*output_ops['meanImg'].shape, 3), dtype='uint8')
        preview[..., 0] = mean1
        if dual_channel:
            mean2 = output_ops['meanImg_chan2'] / output_ops['meanImg_chan2'].max() * 255
            preview[..., 1] = mean2
        imwrite(previewname, preview)

        # remove raw movie
        os.remove(s.procpath + 'suite2p\plane0/data_raw.bin')
        os.rename(s.procpath + 'suite2p\plane0/data.bin',
                  s.procpath + prefix + f'_registered_{s.channelnames[0]}.bin')
        if dual_channel:
            os.remove(s.procpath + 'suite2p\plane0/data_chan2_raw.bin')
            os.rename(s.procpath + 'suite2p\plane0/data_chan2.bin',
                      s.procpath + prefix + f'_registered_{s.channelnames[1]}.bin')
