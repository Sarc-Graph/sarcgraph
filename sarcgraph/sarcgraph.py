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
	def __init__(self, output_dir=None, input_type='video'): #, fully_tracked_frame_ratio=0.75
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
				shutil.rmtree(f'./{self.output_dir}')
			except:
				pass
		Path('./' + f'{self.output_dir}/{data_name}/').mkdir(parents=True, exist_ok=True)
		for i, frame in enumerate(data):
			np.save('./' + f'{self.output_dir}/{data_name}/frame-' + f'{i}'.zfill(5) + '.npy', frame)

	def filter_data(self, data):
		if len(data.shape) != 4 or data.shape[-1] != 1:
			raise ValueError(f"Passed array ({data.shape}) is not of the right shape (frames, dim_1, dim_2, channels=1)")
		filtered_data = np.zeros(data.shape[:-1])
		for i, frame in enumerate(data[:,:,:,0]):
			laplacian = laplace(frame)
			filtered_data[i] = gaussian(laplacian)
		return filtered_data

	def zdisc_info_to_pandas(self, zdiscs_info_all):
		"Create a pandas dataframe that captures the z-discs."
		data_frames = []
		for i, zdiscs_info_frame in enumerate(zdiscs_info_all):
			p1 = zdiscs_info_frame[:,2:4]
			p2 = zdiscs_info_frame[:,4:6]
			fake_mass = 11 * np.sum((p1-p2)**2, axis=1)**2
			frame_id = np.ones((zdiscs_info_frame,1))
			zdisc_id_in_frame = np.arange(0, len(zdiscs_info_frame), 1)
			zdiscs_info_frame_extended = np.hstack((frame_id, zdisc_id_in_frame, zdiscs_info_frame, fake_mass))
			data_frames.append(pd.DataFrame(zdiscs_info_frame_extended, columns=['frame_id', 'zdisc_id', 'x', 'y', 'p1_x', 'p1_y', 'p2_x', 'p2_y', 'mass']))		
		return pd.concat(data_frames)

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
		print('fuck you!')
		self.save_frames(filtered_data, data_name='filtered_data')
		print('fuck you too!')
		return filtered_data

	def zdisc_detection(self, filtered_frames):
		length_checker = np.vectorize(len)
		valid_contours = []
		for i, frame in enumerate(filtered_frames):
			contour_thresh = threshold_otsu(frame)
			contours = measure.find_contours(frame, contour_thresh)
			contours_size = length_checker(contours)
			valid_contours.append(np.delete(contours, np.where(contours_size < 8)[0]))
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
		contours_all = self.zdisc_detection(filtered_data)
		zdiscs_processed_all = []
		for contours_frame in contours_all:
			zdiscs_processed_frame = np.zeros((len(contours_frame), 6))
			for i, contour in enumerate(contours_frame):
				zdiscs_processed_frame[i] = self.zdisc_processing(contour)
			zdiscs_processed_all.append(zdiscs_processed_frame)
		self.save_frames(data=zdiscs_processed_all, data_name='zdiscs-info')
		return zdiscs_processed_all

	##########################################################################################
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	# Track z-discs 
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	##########################################################################################
	def zdisc_tracking(self, tp_depth=4, input_path=None, zdiscs_info=None):
		if self.input_path == 'images':
			print('Cannot perform tracking on a single frame image.')
			return None
		
		if input_path is None and zdiscs_info is None:
			raise ValueError("Either input_path to the original video/image or a numpy array of frame by frame zdiscs_info should be specified..")
		elif zdiscs_info:
			pass
		else:
			zdiscs_info = self.segmentation(input_path)
		
		features = self.zdisc_info_to_pandas(zdiscs_info)
		# Run tracking --> using the trackpy package 
		# http://soft-matter.github.io/trackpy/v0.3.0/tutorial/prediction.html
		t = tp.link_df(features, tp_depth, memory=int(len(zdiscs_info))) 
		t1 = tp.filter_stubs(t, int(len(zdiscs_info)*.10))
		
	def zdisc_tracking(self, tp_depth=4):
		frames = []
		for frame_num in range(self.num_frames):
			frames.append(self.numpy2pandas(frame_num))

		features = pd.concat(frames)
		# Run tracking --> using the trackpy package 
		# http://soft-matter.github.io/trackpy/v0.3.0/tutorial/prediction.html
		t = tp.link_df(features, tp_depth, memory=int()) 
		t1 = tp.filter_stubs(t, int(self.num_frames*.10))
		
		# Extract the results from tracking 
		frame = t1.frame_id.to_numpy()
		xall = t1.x.to_numpy()
		yall = t1.y.to_numpy()
		orig_idx = t1.zdisc_id.to_numpy()
		particle = t1.particle.to_numpy()
		end1_idx1 = t1.p1_x.to_numpy()
		end1_idx2 = t1.p1_y.to_numpy()
		end2_idx1 = t1.p2_x.to_numpy()
		end2_idx2 = t1.p2_y.to_numpy()

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

	"""
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