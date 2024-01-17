import cv2
# from matplotlib import pyplot as plt
import numpy


def pull_motion_energy(vid_fn, output_fn):
    pw_n = 16
    # read input
    im = cv2.VideoCapture(vid_fn)
    ret, frame = im.read()
    fps = im.get(cv2.CAP_PROP_FPS)
    h, w = frame.shape[:2]
    hw = int(w / 2)
    hh = int(h / 2)
    n_frames = int(im.get(cv2.CAP_PROP_FRAME_COUNT))
    pw_res = n_frames / pw_n - 1
    # init output
    gamma = 0.33
    output_gamma = numpy.array([((i / 255.0) ** gamma) * 255 for i in numpy.arange(0, 256)]).astype('uint8')
    mm_trace = numpy.empty(n_frames)

    pw_data = []
    n = 0
    pw_n = 0
    while frame is not None:
        IM = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # keep frames of preview
        if pw_n > pw_res:
            pw_data.append(numpy.copy(IM))
            pw_n = 0
        pw_n += 1
        # compute mm
        # downsample IM
        IM = cv2.resize(IM, (hw, hh), interpolation=cv2.INTER_AREA).astype('float')
        # keep rolling background
        alpha = 0.3
        if n < 1:
            BG = IM
        else:
            BG = alpha * IM + (1 - alpha) * BG
        # rolling absdiff
        D = numpy.abs(IM - BG)
        beta = 0.9
        if n < 1:
            MM = D
        else:
            MM = beta * D + (1 - beta) * MM

        mm_trace[n] = MM.mean()

        n += 1
        ret, frame = im.read()
    mm_trace[0] = mm_trace[1]
    numpy.save(output_fn, mm_trace)

    # # save preview
    # fig_w = 4
    # fig_h = 4  # int(len(pw_data) / fig_w)
    # fig, ax = plt.subplots(nrows=fig_h, ncols=fig_w, figsize=(19, 6), sharey=True, sharex=True)
    # for ca in ax.flat:
    #     ca.spines['right'].set_visible(False)
    #     ca.spines['top'].set_visible(False)
    #     ca.spines['bottom'].set_visible(False)
    #     ca.spines['left'].set_visible(False)
    #     ca.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    #     ca.tick_params(axis='y', which='both', right='off', left='off', labelleft='off')
    #     ca.set_xticks([])
    #     ca.set_yticks([])
    # for ca, Y in zip(ax.flat[:len(pw_data)], pw_data):
    #     ca.imshow(Y, )  # cmap='Greys')
    # plt.tight_layout()
    # fig.savefig(output_fn.replace('.npy', '.png'))
    # plt.close()

    return mm_trace
