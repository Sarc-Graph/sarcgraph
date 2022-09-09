import numpy as np
import matplotlib.pyplot as plt
import imageio

from pathlib import Path

# import cv2


class SarcGraphTools:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        pass

    def visualize_zdiscs(self, frame=0, include_eps=False):
        input_dir = self.input_dir
        """Visualize the results of z-disk segmentation."""
        frame_str = str(frame).zfill(5)
        # load raw image file
        raw_img = np.load(
            f"{input_dir}/raw_frames/frame-{frame_str}.npy", allow_pickle=True
        )

        # load segmented zdisc contours
        contour_list = np.load(
            f"{input_dir}/contours/frame-{frame_str}.npy", allow_pickle=True
        )

        # load z_disc locations
        # z_disc_info = np.load(
        #    f"{input_dir}/zdiscs-info/frame-{frame_str}.npy",
        #    allow_pickle=True
        # )

        """
        # --> import sarcomeres
        sarc_data = np.loadtxt(
            'ALL_MOVIES_PROCESSED/' +
            folder_name +
            '/segmented_sarc/frame-%04d_sarc_data.txt'%(frame)
        )
        sarc_x = sarc_data[:,2]
        sarc_y = sarc_data[:,3]
        """

        ax = plt.axes()
        ax.set_aspect("equal")
        ax.imshow(raw_img[:, :, 0], cmap=plt.cm.gray)
        ax.set_title(f"{len(contour_list)} z-disks found in frame {frame}")
        for contour in contour_list:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])

        # ax2 = plt.axes()
        # ax2.imshow(raw_img, cmap=plt.cm.gray)
        # ax2.set_title(
        #     'sarcomeres -- frame %i, %i found'%(frame,sarc_x.shape[0])
        # )
        # ax2.plot(sarc_y,sarc_x,'r*',markersize=3)
        # ax2.set_xticks([]); axs[1].set_yticks([])

        output_file = f"{input_dir}/visualization/segmentated-zdiscs-frame-{frame_str}"
        Path(f"{input_dir}/visualization/").mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file + ".png", dpi=300, bbox_inches="tight")
        if include_eps:
            plt.savefig(output_file + ".eps", bbox_inches="tight")
        return


###############################################################################
def visualize_contract_anim_movie(
    folder_name,
    re_run_timeseries=False,
    use_re_run_timeseries=False,
    keep_thresh=0.75,
    include_eps=False,
):
    """Visualize detected and tracked sarcomeres."""
    plot_info_frames_fname = (
        "ALL_MOVIES_PROCESSED/"
        + folder_name
        + "/timeseries/"
        + tag_vis
        + "plotting_all_frames.pkl"
    )
    ALL_frames_above_thresh = pickle.load(open(plot_info_frames_fname, "rb"))
    plot_info_x_pos_fname = (
        "ALL_MOVIES_PROCESSED/"
        + folder_name
        + "/timeseries/"
        + tag_vis
        + "plotting_all_x.pkl"
    )
    ALL_x_pos_above_thresh = pickle.load(open(plot_info_x_pos_fname, "rb"))
    plot_info_y_pos_fname = (
        "ALL_MOVIES_PROCESSED/"
        + folder_name
        + "/timeseries/"
        + tag_vis
        + "plotting_all_y.pkl"
    )
    ALL_y_pos_above_thresh = pickle.load(open(plot_info_y_pos_fname, "rb"))
    sarc_data_normalized_fname = (
        "ALL_MOVIES_PROCESSED/"
        + folder_name
        + "/timeseries/"
        + tag_vis
        + "tracking_results_leng.txt"
    )
    all_normalized = np.loadtxt(sarc_data_normalized_fname)

    if use_re_run_timeseries:
        out_plots = out_analysis + "/for_plotting_contract_anim"
    else:
        out_plots = out_analysis + "/contract_anim"

    if not os.path.exists(out_plots):
        os.makedirs(out_plots)

    # --> plot every frame, plot every sarcomere according to normalized fraction length
    color_matrix = np.zeros(all_normalized.shape)
    for kk in range(0, all_normalized.shape[0]):
        for jj in range(0, all_normalized.shape[1]):
            of = all_normalized[kk, jj]
            if of < -0.2:
                color_matrix[kk, jj] = 0
            elif of > 0.2:
                color_matrix[kk, jj] = 1
            else:
                color_matrix[kk, jj] = of * 2.5 + 0.5

    img_list = []
    for t in range(0, num_frames):
        img = get_frame_matrix(folder_name, t)

        plt.figure()
        plt.imshow(img, cmap=plt.cm.gray)
        for kk in range(0, all_normalized.shape[0]):
            if t in ALL_frames_above_thresh[kk]:
                ix = np.argwhere(np.asarray(ALL_frames_above_thresh[kk]) == t)[0][0]
                col = (1 - color_matrix[kk, t], 0, color_matrix[kk, t])
                yy = ALL_y_pos_above_thresh[kk][ix]
                xx = ALL_x_pos_above_thresh[kk][ix]
                plt.scatter(yy, xx, s=15, color=col, marker="o")

        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(out_plots + "/" + file_root + "_length")
        if include_eps:
            plt.savefig(out_plots + "/" + file_root + "_length.eps")
        plt.close()
        img_list.append(imageio.imread(out_plots + "/" + file_root + "_length.png"))

    if num_frames > 1:
        imageio.mimsave(out_plots + "/contract_anim.gif", img_list)
