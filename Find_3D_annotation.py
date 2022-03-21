"""
Author: Christopher Z. Eddy
Contact: eddych@oregonstate.edu
Purpose: Currently, the annotations found in Datasets are only in 2D - that is,
	we have taken cells and done 2D projections for annotations. However, for 3D
	systems, we need to have 3D annotated images. To overcome this barrier, I
	have proposed the following solution.
	(1) For all the X,Y pixels in the cell object, find the mean z-axis profile.
	(2) Fit a Gaussian distribution to the mean z-axis profile to find which
		slices the object is mostly in.
	(3) For each individual X,Y pixel in cell object, find its z-axis profile.
	(4) Multiply the Gaussian fitted distribution by the individual pixel z-axis
		profile (Note, the individual profile may have cells in multiple slices;
		however, by multiplying by the gaussian profile, we only consider a
		finite range of axial pixels that may belong to the object).
	(5) Take x, y, and z pixels as 3D label for cell.

USAGE NOTES:
	****SAVING: I'm not sure if saving the output est. mask data as a mesh of
		vertices and faces is the best strategy. The problem with the vertices is we
		need to recreate the object and manage to place it back into the stack.
		That is not straightforward. Instead I recommend we save the x,y,z pixels into
		a dictionary annotation. There are fewer bytes to store when storing pixels
		than vertices and faces.
	*** Sometimes the interactive plots fail to update correctly (i.e. the
		image slice may update, but the mask slice will not). I have found that
		restarting the script fixes the issue; really not sure on the source of
		the problem.

TEST CASES: Overlapping ellipsoids; Spatially separated ellipsoids. See bottom
			of this worksheet for examples!

REQUIREMENTS: python = 3.7, scipy >= 1.5.4, numpy, matplotlib >= 3.3.2,
			  scikit-image >= 0.16.2, tifffile
"""
##When using MacOS, generating plots of 3D meshes tend to run slow. If you can
##change the backend of matplotlib commented below, it should speed things up.
# import matplotlib
# matplotlib.use('Qt5Agg') #supposed to speed things up, but trouble importing.
import matplotlib.pyplot as plt
import numpy as np
import numpy
from scipy.optimize import curve_fit
import skimage.measure
from scipy.signal import argrelextrema
from CellDataset import *

class Find_3D(object):
	def __init__(self):
		pass

	def gauss(self, x, *p):
		A, mu, sigma = p
		return A*numpy.exp(-(x-mu)**2/(2.*sigma**2))

	def run_fit(self, dta, p0, x=None):
		"""
		INPUTS
		-----------------------------------------------------
		dta: 1-D np array or list vector of axial intensity from image stack

		p0: List, initial estimates for approximating Gaussian fit on dta.
			The initial guess for the fitting coefficients (A, mu and sigma)

		x: Optional, array or list (if doing fit on truncated 'dta' for
		   estimated parameters, then pass dta as the argument for x in order
		   to acquire the Gaussian calculated at all the x points in dta)

		OUTPUTS
		------------------------------------------------------
		Plot of labeled profiles

		PURPOSE
		------------------------------------------------------
		Fits test_data vector with a Gaussian using initial estimates p0
		"""
		#dta should be a list
		if not isinstance(dta, list):
			#convert to list
			dta = list(dta)
		coeff, var_matrix = curve_fit(self.gauss, [-1]+list(range(len(dta)))+[len(dta)], [0]+dta+[0], p0=p0)
		if x is None:
			# Get the fitted curve
			hist_fit = self.gauss(list(range(len(dta))), *coeff)
		else:
			# If the fit above was over a truncated form of dta, we want the
			# fit estimated at each point in the full form of dta.
			hist_fit = self.gauss(list(range(len(x))), *coeff)
		return hist_fit

	def plot_fit(self, test_data, fit_data):
		"""
		INPUTS
		-----------------------------------------------------
		test_data: list vector of axial intensity profile.

		fit_data: list vector of fitted axial intensity profile

		OUTPUTS
		------------------------------------------------------
		Plot of labeled profiles

		PURPOSE
		------------------------------------------------------
		Compare the intensity profile with the Gaussian fit to verify estimation
		"""
		plt.plot(list(range(len(test_data))), test_data, label='Test data')
		plt.plot(list(range(len(fit_data))), fit_data, label='Fitted data')
		plt.show()

	def run_analysis(self, mask, IM, return_verts = False, Zall = None):
		"""
		INPUTS
		-----------------------------------------------------
		mask = 3D np.array with shape H x W x N where N are the number of objects
			each channel contains just a single object.

		IM = 4D np.array Z-stack with shape H x W x Z x 1 or 3D np.array (HxWxZ)

		Zall = optional argument, must be list of floats, integers, or None if
			provided. Zall is the approximate axial location of the cell,
			helpful if there are overlapping cells in the image. It is not
			recommended to provide Zall, use only if necessary.
			Please note the predefined limited fitting range of +- 4 slices if
			Zall is provided. See code below.

		OUTPUTS
		------------------------------------------------------
		Iall = numpy array, H x W x Z estimated binary image stack of IM

		obj_pixels = list of object arrays containing object pixel locations.

		PURPOSE
		------------------------------------------------------
		Takes 2D annotations of 3D objects (mask) and, using the raw image
		z-stack, maps the 2D pixels of the object to form 3D annotations.

		METHODS
		------------------------------------------------------
		Takes the axial profile of the average intensity of all x,y pixels in
		object and forms an object-gaussian profile. Then, we take the axial
		profile of each individual pixel in object and multiply it by the
		object-gaussian profile, which acts to suppress multiple peaks in the
		individual pixel profile observed when other objects are in different
		Z locations. Then, each x-y pixel is assigned with z pixel(s) based on
		the multiplied profile. A full 3D annotation is achieved once all x-y
		pixels belonging to object have been analyzed.
		"""
		if len(IM.shape)<4:
			IM = np.expand_dims(IM,axis=-1) #just for how the code runs.

		if Zall is not None:
			assert isinstance(Zall, list), "Argument 'Zall' must be a list type containing floats, integers, or None types."
		##option to store the vertices and faces, which are useful for cloudpoint and
		##useful for 3D reconstructions when using good axial resolution.
		if return_verts:
			obj_verts = []
			obj_faces = []
			obj_norms = []
		##option to store object pixels. This may slow things down, but the data storage
		##is undeniably less than the vertices and faces. More efficient to do
		##construction AFTER reloading than to do it and save it here.
		obj_pixels=[]
		Iall = np.zeros_like(IM).squeeze()
		for obj in range(mask.shape[2]):

			obj_pixels.append([])
			XY = np.argwhere(mask[:,:,obj])
			#intensity profile of full object image Pixels through z
			I_proj = [np.mean([IM[X,Y,Z,0] for (X,Y) in XY]) for Z in range(IM.shape[2])]
			#Fit the profile with Gaussian.

			# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
			if Zall is None:
				p0 = [np.max(I_proj), np.argmax(I_proj), 1.]
				try:
					#Fit the data with a single Gaussian
					hist_fit = self.run_fit(I_proj, p0)
				except:
					print("Can't find optimal params.")
					#generally happens when the cell is located on the ends.
					import pdb;pdb.set_trace()
			else:
				Z = Zall[obj]
				if Z is None:
					#use default
					p0 = [np.max(I_proj), np.argmax(I_proj), 1.]
					try:
						#Fit the data with a single Gaussian
						hist_fit = self.run_fit(I_proj, p0)
					except:
						print("Can't find optimal params.")
						#generally happens when the cell is located on the ends.
						import pdb;pdb.set_trace()
				else:
					#find the approximate max near z. This is very, very important
					#find approximate axial-location of local maxima in C
					localmaxs = argrelextrema(np.array(I_proj),np.greater)[0]
					#find the closest to
					X = localmaxs[np.argmin(np.abs(localmaxs - float(Z)))]
					p0 = [I_proj[X], X, 1.]
					#print("You should check that the fits are good at this step between hist_fit and I_proj.")
					#import pdb;pdb.set_trace()
					try:
						hist_fit = self.run_fit([0]*max([0,X-5]) + I_proj[max([0,X-5]):min([len(I_proj),X+5])], p0, x=I_proj)
					except:
						print("Can't find optimal params.")
						#generally happens when the cell is located on the ends.
						import pdb;pdb.set_trace()

			if return_verts:
				Ifilled = np.zeros_like(IM).squeeze()
			#for each x,y pixel in object, find which axial (z) pixels belong to it
			for pix in range(XY.shape[0]):
				##take axial profile of pixel
				s_proj=[IM[XY[pix][0],XY[pix][1],Z,0] for Z in range(IM.shape[2])]
				##multiply axial profile of pixel by entire object profile
				d_proj = [s_proj[i]*hist_fit[i] for i in range(len(s_proj))]
				##define a threshold for axial pixels that belong to object.
				zgood = np.argwhere(np.array(d_proj)>=np.max(d_proj)/10).flatten()
				##The above threshold is kind of arbitrary, but it is necessary to define a cutoff.

				##save 3D pixel locations into object array.
				if len(obj_pixels[obj])>0:
					#if list is not empty
					obj_pixels[obj][0] = np.concatenate([obj_pixels[obj][0],np.concatenate([np.repeat(XY[pix,:][None,...],len(zgood),axis=0),zgood.reshape(len(zgood),1)],axis=1)],axis=0)
				else:
					#if list is empty
					obj_pixels[obj].append(np.concatenate([np.repeat(XY[pix,:][None,...],len(zgood),axis=0),zgood.reshape(len(zgood),1)],axis=1))
				for z in zgood:
					if return_verts:
						Ifilled[XY[pix][0],XY[pix][1],z]=1.
					Iall[XY[pix][0],XY[pix][1],z]=1.
			if return_verts:
				##Find vertices of object for 3D cloud or reconstruction.
				#verts, faces, normals, values
				try:
					verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(Ifilled, 0.0)
					obj_verts.append(verts)
					obj_faces.append(faces)
					obj_norms.append(normals)
				except:
					print("problem with threshold defined in zgood; zgood is empty.")
					import pdb;pdb.set_trace()

		obj_pixels = [x[0] for x in obj_pixels] #eliminate redundant lists in list.

		if return_verts:
			return obj_verts, obj_faces, obj_norms, obj_pixels, Iall
		else:
			return obj_pixels, Iall.astype(np.bool)

	def save_zstack(self, data, filepath=None):
		"""
		INPUTS
		-----------------------------------------------------
		data = 3D np array shape H x W x Z
			data should be the output 'Iall' from run_analysis

		filepath = string, filepath with filename and ending '.ome.tif'

		OUTPUTS
		------------------------------------------------------
		Saves a .ome.tif file to the home directory or directory specified in
		filepath
		"""
		#data should be shape
		A = np.rollaxis(data,0,3)
		A = np.rollaxis(A,0,3)
		import tifffile
		if filepath is None:
			tifffile.imwrite("test.ome.tif",A)
		else:
			tifffile.imwrite(filepath,A)

	def plot_3d_mesh(self, verts, faces):
		"""
		INPUTS
		-----------------------------------------------------
		verts = list of vertices from run_analysis

		faces = list of faces from run_analysis

		OUTPUTS
		------------------------------------------------------
		Plot of 3D mesh object. Runs slow [many, many faces is the problem].
		"""
		from mpl_toolkits.mplot3d import Axes3D
		from mpl_toolkits.mplot3d.art3d import Poly3DCollection
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.add_collection3d(Poly3DCollection(verts[faces]))
		xy_grace=5
		z_grace=1
		ax.set_ylim(np.min(verts[:,1])-xy_grace,np.max(verts[:,1])+xy_grace)
		ax.set_xlim(np.min(verts[:,0])-xy_grace,np.max(verts[:,0])+xy_grace)
		ax.set_zlim(np.min(verts[:,2])-z_grace,np.max(verts[:,2])+z_grace)
		plt.show()

	def compare_z_stacks(self, IM, M):
		"""
		INPUTS
		-----------------------------------------------------
		IM = 3D np array with shape H x W x Z
			Z-stack of images

		M = 3D np array with shape H x W x Z
			M is output or "Iall" from run_analysis with shape HxWxZ

		OUTPUTS
		------------------------------------------------------
		Interactive plot of z-stack with estimated object masks.

		"""
		from matplotlib.widgets import Slider, Button
		import copy
		# Create the figure and the line that we will manipulate
		fig, (ax1, ax2) = plt.subplots(1,2)
		shown_image = ax1.imshow(IM[:,:,IM.shape[2]//2])
		shown_mask = ax2.imshow(M[:,:,M.shape[2]//2])
		ax1.set_title('Z-stack')
		ax2.set_title('Estimated Masks')
		#use shown_image.set_data(images[z])
		# adjust the main plot to make room for the sliders
		plt.subplots_adjust(bottom=0.25)
		# Make a horizontal slider to control the initial padding
		axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
		# define the values to use for snapping
		allowed_zs = np.array(list(range(IM.shape[2])))
		slider_handle = Slider(ax=axfreq,label='Z', valmin=0, \
			valmax=IM.shape[2]-1, valinit=IM.shape[2]//2, valstep=1)

		def update(val):
			v = slider_handle.val
			if isinstance(v,int):
				shown_image.set_data(IM[:,:,v])
				shown_mask.set_data(M[:,:,v])
			else:
				shown_image.set_data(IM[:,:,v.astype(int)])
				shown_mask.set_data(M[:,:,v.astype(int)])
			fig.canvas.draw_idle()

		slider_handle.on_changed(update)

		ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
		ax_save = plt.axes([0.6, 0.025, 0.15, 0.04])
		button = Button(ax_reset, 'Reset', hovercolor='0.975')
		button_close = Button(ax_save, 'Close', hovercolor='0.975')

		def reset(event):
			slider_handle.reset()

		def close(event):
			plt.close()

		button.on_clicked(reset)
		button_close.on_clicked(close)
		plt.show()

	def plot_z_stack(self, IM):
		"""
		INPUTS
		-----------------------------------------------------
		IM = 3D np array with shape H x W x Z
			Z-stack of images

		OUTPUTS
		------------------------------------------------------
		Interactive plot of z-stack.

		"""
		from matplotlib.widgets import Slider, Button
		import copy
		# Create the figure and the line that we will manipulate
		fig, ax1 = plt.subplots(1)
		shown_image = ax1.imshow(IM[:,:,IM.shape[2]//2])
		ax1.set_title('Z-stack')
		#use shown_image.set_data(images[z])
		# adjust the main plot to make room for the sliders
		plt.subplots_adjust(bottom=0.25)
		# Make a horizontal slider to control the initial padding
		axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
		# define the values to use for snapping
		allowed_zs = np.array(list(range(IM.shape[2])))
		slider_handle = Slider(ax=axfreq,label='Z', valmin=0, \
			valmax=IM.shape[2]-1, valinit=IM.shape[2]//2, valstep=1)

		def update(val):
			v = slider_handle.val
			if isinstance(v,int):
				shown_image.set_data(IM[:,:,v])
			else:
				shown_image.set_data(IM[:,:,v.astype(int)])
			fig.canvas.draw_idle()

		slider_handle.on_changed(update)

		ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
		ax_save = plt.axes([0.6, 0.025, 0.15, 0.04])
		button = Button(ax_reset, 'Reset', hovercolor='0.975')
		button_close = Button(ax_save, 'Close', hovercolor='0.975')

		def reset(event):
			slider_handle.reset()

		def close(event):
			plt.close()

		button.on_clicked(reset)
		button_close.on_clicked(close)
		plt.show()

	def generate_test_mask(self, nearby_xy = False, stacked = False, angled = False, overlapped = True):
		#ellipsoids
		x = np.linspace(0,127,128)
		y = np.linspace(0,127,128)
		z = np.linspace(0,24,25)
		u,_,_ = np.meshgrid(x,y,z) #x,y,z coordinates. H x W x Z matrix.
		mask = np.zeros_like(u)
		if nearby_xy:
			"""
			Put ellipsoids in the center stack but different x y locations
			"""
			a, b, c = 4, 2, 1
			r=5
			xc = len(x)//4
			xcc = len(x)*3//4
			zc = len(z)//2
			mask_gt = np.zeros(shape=(len(x),len(y),2))
			for i in range(len(x)):
				for j in range(len(y)):
					for k in range(len(z)):
						if ((x[i]-xc)*(x[i]-xc))/(a*a) + ((y[j]-xc)*(y[j]-xc))/(b*b) + ((z[k]-zc)*(z[k]-zc))/(c*c) <= r*r:
							mask[i,j,k]=1.
							mask_gt[i,j,0]=1.
						if ((x[i]-xcc)*(x[i]-xcc))/(a*a) + ((y[j]-xcc)*(y[j]-xcc))/(b*b) + ((z[k]-zc)*(z[k]-zc))/(c*c) <= r*r:
							mask[i,j,k]=1.
							mask_gt[i,j,1]=1.
		elif stacked:
			"""
			Stack ellipsoids perfectly above and below.
			"""
			a, b, c = 4, 2, 1
			r=6
			xc = len(x)//2
			zc = len(z)//4 #THERE IS SOME ISSUE WITH SETTING THE Z center NOT IN THE CENTER...
			zcc = len(z)*3//4
			mask_gt = np.zeros(shape=(len(x),len(y),2))
			for i in range(len(x)):
				for j in range(len(y)):
					for k in range(len(z)):
						if ((x[i]-xc)*(x[i]-xc))/(a*a) + ((y[j]-xc)*(y[j]-xc))/(b*b) + ((z[k]-zc)*(z[k]-zc))/(c*c) <= r*r:
							mask[i,j,k]=1.
							mask_gt[i,j,0]=1.
						if ((x[i]-xc)*(x[i]-xc))/(a*a) + ((y[j]-xc)*(y[j]-xc))/(b*b) + ((z[k]-zcc)*(z[k]-zcc))/(c*c) <= r*r:
							mask[i,j,k]=1.
							mask_gt[i,j,1]=1.
		elif angled:
			"""
			Single ellipsoid at an angle
			Not yet coded.
			"""
			mask_gt = np.zeros(shape=(len(x),len(y),1))
			print("'angled' argument option has not yet been programmed.")

		elif overlapped:
			"""
			Stack ellipsoids but make them oriented differently.
			"""
			a1, b1, c1 = 4, 2, 1
			a2, b2, c2 = 2, 4, 1
			mask_gt = np.zeros(shape=(len(x),len(y),2))
			r=6
			xc = len(x)//2
			zc = len(z)//4 #THERE IS SOME ISSUE WITH SETTING THE Z center NOT IN THE CENTER...
			zcc = len(z)*3//4
			for i in range(len(x)):
				for j in range(len(y)):
					for k in range(len(z)):
						if ((x[i]-xc)*(x[i]-xc))/(a1*a1) + ((y[j]-xc)*(y[j]-xc))/(b1*b1) + ((z[k]-zc)*(z[k]-zc))/(c1*c1) <= r*r:
							mask[i,j,k]=1.
							mask_gt[i,j,0]=1.
						if ((x[i]-xc)*(x[i]-xc))/(a2*a2) + ((y[j]-xc)*(y[j]-xc))/(b2*b2) + ((z[k]-zcc)*(z[k]-zcc))/(c2*c2) <= r*r:
							mask[i,j,k]=1.
							mask_gt[i,j,1]=1.
		#self.plot_z_stack(mask)
		return mask.astype(np.bool), mask_gt.astype(np.bool)

	def generate_test_image(self, mask):
		"""
		Create an intensity image to accompany mask
		"""
		sig_obj = 0.2
		mu_obj = 0.6
		sig_bg = 0.01
		mu_bg = 0.05
		#draw from normal distributions.
		IM = np.zeros(shape=mask.shape, dtype=np.float32)
		obj = np.random.normal(mu_obj, sig_obj, mask.shape)
		bg = np.random.normal(mu_bg, sig_bg, mask.shape)
		IM = np.where(mask, obj, bg)
		IM = np.clip(IM, 0., 1.)
		return IM

"""
Working Examples commented below!
"""
## TEST CASE of two ellipsoids.
# F = Find_3D()
# #generate test mask with 2 ellipsoids that are overlapping, but oriented differently.
# M, Mgt = F.generate_test_mask(nearby_xy = False, stacked = False, angled = False, overlapped = True)
# #M is the ground truth Z-stack (H x W x Z). Mgt is the maximum projected version
# # with one object in each channel (H x W x 2).
# #Generate test image with noise.
# IM = F.generate_test_image(M)
# #compute 3D mask estimation from 2D max projections without providing a Z guess of cell locations.
# _, Mtest = F.run_analysis(Mgt, IM)
# #compute 3D mask estimation but this time provide guesses for cell Z slice locations.
# _, Mbetter = F.run_analysis(Mgt, IM, Zall = [5, 17])
#
# #compare ground truth 3D mask and produced 3D estimated annotations.
# F.compare_z_stacks(M, Mbetter)
# #often, pixels of Mbetter are missing due to the added noise in the generated image.
# #compare image and 3D estimated annotations.
# F.compare_z_stacks(IM, Mbetter)


##With CellDataset
# CP = CellDataset(dataset_path="/Users/czeddy/Documents/Auto_Seg/CellAnnotate/CellAnnotate/datasets/example")
# CP.run_prep()
# M, IM, _ = CP.load_image_gt(0)
##import Find_3D_annotation
# F = Find_3D()
# O, Iall = F.run_analysis(M, IM)
# F.compare_z_stacks(np.squeeze(IM),Iall)

## With CellPose Network
# CP = CellPose(mode="training",dataset_path="/users/czeddy/documents/auto_seg/datasets/v7_mini",data_type="Cell3D")
# CP.import_train_val_data()
# mask = CP.dataset_train.load_mask(0)
# IM = CP.dataset_train.load_image(0, CP.config.INPUT_DIM, mask=mask)
##import Find_3D_annotation
# F = Find_3D()
# O, Iall = F.run_analysis(mask, IM)
# F.compare_z_stacks(np.squeeze(IM),Iall)
