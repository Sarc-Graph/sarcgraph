import shutil
import numpy as np
import pandas as pd
import trackpy as tp

import skvideo.io
import skimage.io
import skvideo.utils

from skimage.filters import laplace, gaussian, threshold_otsu
from skimage import measure
from scipy.spatial import distance_matrix
from collections import Counter
from pathlib import Path

##########################################################################################
# Input info and sest up 
##########################################################################################
class SarcGraph:
	def __init__(self, output_dir=None, input_type='video'): #tp_depth=4, fully_tracked_frame_ratio=0.75
		if output_dir == None:
			raise ValueError('Output directory should be specified.')
		self.output_dir = output_dir
		self.input_type = input_type

		"""
		frames_path = f'ALL_MOVIES_MATRICES/{self.folder_name}/*'
		self.num_frames = len(glob.glob(f'{frames_path}.npy'))
		if self.num_frames == 0:
			self.num_frames = len(glob.glob(f'{frames_path}.txt'))
		
		external_folder_name = 'ALL_MOVIES_PROCESSED'
		self.out_bands = f'{external_folder_name}/{self.folder_name}/segmented_bands'
		self.out_sarc = f'{external_folder_name}/{self.folder_name}/segmented_sarc'
		self.out_track = f'{external_folder_name}/{self.folder_name}/tracking_results'
		Path(self.out_bands).mkdir(parents=True, exist_ok=True)
		Path(self.out_sarc).mkdir(parents=True, exist_ok=True)
		Path(self.out_track).mkdir(parents=True, exist_ok=True)

		self.tracked_discs_file = f'{self.out_track}/tracking_results_zdiscs.txt'
		"""

	##########################################################################################
	# Utilities
	##########################################################################################
	def data_loader(self, input_path):
		if self.input_type == 'video':
			data = skvideo.io.vread(input_path)
			if data.shape[0] > 1:
				return data
		return skimage.io.imread(input_path)
		
	def _to_gray(self, data):
		return skvideo.utils.rgb2gray(data)
	
	def save_frames(self, data, data_name, del_existing=True):
		if del_existing:
			try:
				shutil.rmtree(f'{self.output_dir}')
			except:
				pass
		Path(f'{self.output_dir}/{data_name}/').mkdir(parents=True, exist_ok=True)
		for i, frame in enumerate(data):
			np.save(f'{self.output_dir}/{data_name}/frame-' + f'{i}'.zfill(5) + '.npy', frame)

	def filter_data(self, data):
		if len(data.shape) != 4 or data.shape[-1] != 1:
			raise ValueError(f"Passed array ({data.shape}) is not of the right shape (frames, dim_1, dim_2, channels=1)")
		filtered_data = np.zeros(data.shape[:-1])
		for i, frame in enumerate(data[:,:,:,0]):
			laplacian = laplace(frame)
			filtered_data[i] = gaussian(laplacian)
		return filtered_data

	"""
	##########################################################################################
	def numpy2pandas(self, frame):
		"Create a pandas dataframe that captures the z-discs."
		file_root = self.get_frame_name(frame)
		filename = f'ALL_MOVIES_PROCESSED/{self.folder_name}/segmented_bands/{file_root}_bands.txt'
		numpy_file = np.loadtxt(filename)
		cent_idx1 = numpy_file[:,0]
		cent_idx2 = numpy_file[:,1]
		end1_idx1 = numpy_file[:,2]
		end1_idx2 = numpy_file[:,3]
		end2_idx1 = numpy_file[:,4]
		end2_idx2 = numpy_file[:,5]
		fake_mass = ((end1_idx1 - end2_idx1)**2.0 + (end1_idx2 - end2_idx2)**2.0 )**2.0 * 11
		orig_idx_all = np.arange(0, numpy_file.shape[0], 1)
		pd_dataframe = pd.DataFrame(dict(y=cent_idx2, x=cent_idx1, orig_idx=orig_idx_all, end1_idx1=end1_idx1, end1_idx2=end1_idx2, end2_idx1=end2_idx1, end2_idx2=end2_idx2, mass=fake_mass, frame=frame))

		return pd_dataframe

	##########################################################################################
	def compute_length_from_contours(self, cont1, cont2):
		"Compute the length between two z discs from two contours"
		c1_x = np.mean(cont1[:,0])
		c1_y = np.mean(cont1[:,1])
		c2_x = np.mean(cont2[:,0])
		c2_y = np.mean(cont2[:,1])

		return ((c1_x - c2_x)**2.0 + (c1_y - c2_y)**2.0)**0.5
	
	##########################################################################################
	def find_fully_tracked(self):
		tracked_zdiscs = np.loadtxt(self.tracked_discs_file)[:,(0,2,3,4,5,6,7,8)]
		self.num_frames = np.max(tracked_zdiscs[:,0]) + 1

		labels = tracked_zdiscs[:, 1].astype(int)
		unique_labels = np.unique(labels)
		self.fully_tracked_zdiscs = np.empty((0, tracked_zdiscs.shape[1]))
		self.fully_tracked_clusters = np.empty((0, 2))
		self.fully_tracked_label = 0
		self.partially_tracked_zdiscs = np.empty((0, tracked_zdiscs.shape[1]))
		self.partially_tracked_clusters = np.empty((0, 2))
		self.partially_tracked_label = 0
		for lbl in unique_labels:
			index = np.where(labels==lbl)[0]
			zdiscs = tracked_zdiscs[index]
			num_zdiscs = len(zdiscs)
			if num_zdiscs == self.num_frames:    
				zdiscs[:,1] = self.fully_tracked_label
				self.fully_tracked_zdiscs = np.vstack((self.fully_tracked_zdiscs, zdiscs))
				self.fully_tracked_label += 1
			else:
				zdiscs[:,1] = self.partially_tracked_label
				self.partially_tracked_zdiscs = np.vstack((self.partially_tracked_zdiscs, zdiscs))
				self.partially_tracked_clusters = np.vstack((self.partially_tracked_clusters, np.mean(zdiscs[:,(2,3)], axis=0)))
				self.partially_tracked_label += 1
	"""

	##########################################################################################
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	# Segmentation
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	##########################################################################################
	def preprocessing(self, input_path):
		raw_data = self.data_loader(input_path)
		raw_data_gray = self._to_gray(raw_data)
		filtered_data = self.filter_data(raw_data_gray)
		self.save_frames(raw_data_gray, data_name='raw_data_gray_scale')
		self.save_frames(filtered_data, data_name='filtered_data')
		return filtered_data

	def zdisc_detection(self, filtered_frames):
		length_checker = np.vectorize(len)
		for i, frame in enumerate(filtered_frames):
			contour_thresh = threshold_otsu(frame)
			contours = measure.find_contours(frame, contour_thresh)
			contours_size = length_checker(contours)
			valid_contours = np.delete(contours, np.where(contours_size < 8)[0])
		self.save_frames(data=valid_contours, data_name='contours')
		return valid_contours
	
	def zdisc_processing(self, contour):
		"""Process the contour and return important properties. Units of pixels."""
		# coordinates of the center of a contour
		center_coords = np.mean(contour, axis=0)
		# find zdisc direction by matching furthest points on the contour
		dist_mat = distance_matrix(contour, contour)
		indices = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
		# coordinates of the two points on the contour with maximum distance
		p1, p2 = contour[indices[0]], contour[indices[1]]
		return np.hstack((center_coords, p1, p2))

	def segmentation(self, input_path):
		filtered_data = self.preprocessing(input_path)
		contours = self.zdisc_detection(filtered_data)
		zdiscs_processed = np.zeros((len(contours), 6))
		for i, contour in enumerate(contours):
			zdiscs_processed[i] = self.zdisc_processing(contour)
		np.save(f'{self.output_dir}/zdiscs-info.npy', zdiscs_processed)

"""
	##########################################################################################
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	# Track z-discs 
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	##########################################################################################
	# load all of the segmented z-discs into a features array 
	def track_zdiscs(self):
		frames = []
		for frame_num in range(self.num_frames):
			frames.append(self.numpy2pandas(frame_num))

		features = pd.concat(frames)
		# Run tracking --> using the trackpy package 
		# http://soft-matter.github.io/trackpy/v0.3.0/tutorial/prediction.html
		t = tp.link_df(features, self.tp_depth, memory=int(self.num_frames)) 
		t1 = tp.filter_stubs(t, int(self.num_frames*.10))
		
		# Extract the results from tracking 
		frame = t1.frame.to_numpy()
		xall = t1.x.to_numpy()
		yall = t1.y.to_numpy()
		orig_idx = t1.orig_idx.to_numpy()
		particle = t1.particle.to_numpy()
		end1_idx1 = t1.end1_idx1.to_numpy()
		end1_idx2 = t1.end1_idx2.to_numpy()
		end2_idx1 = t1.end2_idx1.to_numpy()
		end2_idx2 = t1.end2_idx2.to_numpy()

		save_data = np.zeros((frame.shape[0],9))
		save_data[:,0] = frame
		save_data[:,1] = orig_idx
		save_data[:,2] = particle
		save_data[:,3] = xall
		save_data[:,4] = yall 
		save_data[:,5] = end1_idx1
		save_data[:,6] = end1_idx2
		save_data[:,7] = end2_idx1
		save_data[:,8] = end2_idx2

		np.savetxt(f'{self.out_track}/tracking_results_zdiscs.txt', save_data)

	##########################################################################################
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	# Merge tracked z-discs 
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	##########################################################################################
	# load all of tracked z-disc clusters and merge similar clusters of partially tracked z-discs
	def merge_zdiscs(self):
		self.find_fully_tracked()

		# connect neighboring clusters
		clusters_dist = distance_matrix(self.partially_tracked_clusters, self.partially_tracked_clusters)
		min_dist = 0.01*np.max(clusters_dist)
		merged_labels = connected_components(clusters_dist < min_dist)[1]
		merged_unq_labels = np.unique(merged_labels)

		# find which new clusters form a complete cluster
		labels = self.partially_tracked_zdiscs[:,1]
		unq_labels = np.unique(labels)
		for lbl in merged_unq_labels:
			merged_clusters_labels = unq_labels[merged_labels==lbl]
			merged_clusters_indices = np.where(merged_labels==lbl)[0].astype(int)
			merged_zdiscs_indices = np.isin(labels, merged_clusters_labels)
			zdiscs = self.partially_tracked_zdiscs[merged_zdiscs_indices]
			
			# if there is some zdiscs from the same frame exist in a set of matched clusters, merge them into one by taking the average of their attributes
			unq_frames = np.unique(zdiscs[:,0]).astype(int)
			for f in unq_frames:
				indices = np.where(zdiscs[:,0]==f)[0]
				if len(indices) > 1:
					zdiscs[indices[0]] = np.mean(zdiscs[indices], axis=0)
					zdiscs = np.delete(zdiscs, indices[1:], axis=0)
			
			num_zdiscs = len(zdiscs)
			if num_zdiscs >= self.frame_threshod * self.num_frames: # if the new merged cells have enough frames
				# add the new merged clusters to the set of complete clusters
				zdiscs[:,1] = self.fully_tracked_label
				self.fully_tracked_zdiscs = np.vstack((self.fully_tracked_zdiscs, zdiscs))
				self.fully_tracked_label += 1
"""