import numpy as np
import matplotlib.pyplot as plt

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
        ax.imshow(raw_img[:, :, 0], cmap=plt.cm.gray)
        ax.set_title(f"{len(contour_list)} z-disks found in frame {frame}")
        for contour in contour_list:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])

        # axs[1].imshow(raw_img, cmap=plt.cm.gray)
        # axs[1].set_title(
        #     'sarcomeres -- frame %i, %i found'%(frame,sarc_x.shape[0])
        # )
        # axs[1].plot(sarc_y,sarc_x,'r*',markersize=3)
        # axs[1].set_xticks([]); axs[1].set_yticks([])

        output_file = f"{input_dir}/visualization/segmentated-zdiscs-frame-{frame_str}"
        Path(f"{input_dir}/visualization/").mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file + ".png", dpi=300, bbox_inches="tight")
        if include_eps:
            plt.savefig(output_file + ".eps", bbox_inches="tight")
        return
