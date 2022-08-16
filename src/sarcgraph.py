import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
import cv2
from skimage.filters import threshold_otsu
from skimage import measure
from scipy import ndimage
from scipy.spatial import distance_matrix
from collections import Counter
import pickle
import glob
from pathlib import Path
from scipy.sparse.csgraph import connected_components

import time

##########################################################################################
# Input info and set up 
##########################################################################################
class SarcGraph:
	def __init__(self, folder_name, gaussian_filter_size=1, tp_depth=4, fully_tracked_frame_ratio=0.75):
		self.folder_name = folder_name
		self.gaussian_filter_size = gaussian_filter_size
		self.tp_depth = tp_depth
		self.frame_threshod = fully_tracked_frame_ratio
	
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

		self.tracked_discs_file = f'{self.out_track}/tracking_results_zdisks.txt'
	
	##########################################################################################
	# Helper functions
	##########################################################################################
	def get_frame_name(self, frame_num):
		if frame_num < 10: file_root = f'frame-000{frame_num}'
		elif frame_num < 100: file_root = f'frame-00{frame_num}'
		else: file_root = f'frame-0{frame_num}'

		return file_root

	##########################################################################################
	def numpy2pandas(self, frame):
		"""Create a pandas dataframe that captures the z-disks."""
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
	def frame2array(self, frame_num):
		"""Get the npy matrix for a frame of the movie."""
		external_folder_name = f'ALL_MOVIES_MATRICES/{self.folder_name}/'
		
		file_root = self.get_frame_name(frame_num)
		root_npy = f'{external_folder_name}{file_root}.npy'
		root_txt = f'{external_folder_name}{file_root}.txt'
		try:
			raw_img = np.load(root_npy)
		except:
			raw_img = np.loadtxt(root_txt)

		return file_root, raw_img

	##########################################################################################
	def process_band(self, cont):
		"""Process the contour and return important properties. Units of pixels."""
		# coordinate 1 of center 
		center_idx1 = np.mean(cont[:,0])
		# coordinate 2 of center 
		center_idx2 = np.mean(cont[:,1])
		# find the maximum distance between points in the contour and identify the coordinates
		dist_mat = distance_matrix(cont,cont)

		args = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
		idx_kk = args[0]
		idx_jj = args[1]
		# identify end_1 and end_2 -- coordinates of the ends 
		end_1x, end_1y = cont[idx_kk,0], cont[idx_kk,1]
		end_2x, end_2y = cont[idx_jj,0], cont[idx_jj,1]

		info = np.array([[center_idx1, center_idx2, end_1x, end_1y, end_2x, end_2y]])

		return info

	##########################################################################################
	def compute_length_from_contours(self, cont1, cont2):
		"""Compute the length between two z disks from two contours"""
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

	##########################################################################################
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	# Segment z-disks 
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	##########################################################################################
	def segment_zdiscs(self):
		start_time = time.time()
		for frame in range(self.num_frames):
			file_root, raw_img = self.frame2array(frame)
		
			# compute the laplacian of the image and then use that to find the contours
			laplacian = cv2.Laplacian(raw_img, cv2.CV_64F)
			laplacian = ndimage.gaussian_filter(laplacian, self.gaussian_filter_size)
			contour_thresh = threshold_otsu(laplacian)
			contour_image = laplacian
			contours =  measure.find_contours(contour_image, contour_thresh)
		
			# create a list of all contours large enough (in pixels) to surround an area
			contour_list = [] 
			for contour in contours:
				if contour.shape[0] >= 8:
					contour_list.append(contour)
			
			# compute properties of each contour i.e. z disk
			num_countors = len(contour_list)
			info = np.zeros((num_countors, 6))
			for i, contour in enumerate(contour_list):
				info[i] = self.process_band(contour)

			# save info per band: center x, center y, end_1x,y, end_2 x,y
			np.savetxt(f'{self.out_bands}/{file_root}_bands.txt', info)
		
			# save contour_list --> pickle the file to come back to later 
			with open(f'{self.out_bands}/{file_root}_raw_contours.pkl', 'wb') as f:
				pickle.dump(contour_list, f)
		print("--- %s seconds ---" % (time.time() - start_time))

	##########################################################################################
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	# Track z-disks 
	# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ #
	##########################################################################################
	# load all of the segmented z-disks into a features array 
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

		np.savetxt(f'{self.out_track}/tracking_results_zdisks.txt', save_data)

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
