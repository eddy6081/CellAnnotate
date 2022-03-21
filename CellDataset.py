import numpy as np
import skimage.io
import skimage.draw
import scipy.stats

class CellDataset(object):

	def __init__(self, dataset_path, dimensionality="3D"):
		"""
		dataset_path:
			Root directory of the dataset. Training MUST have form:
				dataset_directory
				----/directory
				--------/images
				--------/gt
		"""
		if dataset_path[-1]=="/":
			dataset_path=dataset_path[0:-1]

		print("Dataset directory: ", dataset_path)
		self.dataset_dir = dataset_path

		self.dimensionality=dimensionality

	def load_cell(self, dataset_dir, subset=None):
		"""Load a subset of the cell dataset.

		dataset_dir: Root directory of the dataset
		subset: Subset to load. Either the name of the sub-directory,
				such as stage1_train, stage1_test, ...etc. or, one of:
				* train: stage1_train excluding validation images
				* val: validation images from VAL_IMAGE_IDS
		"""
		# Add classes. We have one class.
		# Naming the dataset nucleus, and the class nucleus
		self.add_class("cell", 1, "cell")

		load_dir = os.path.join(dataset_dir, "images")
		# Get image ids from directory names
		image_ids = os.listdir(load_dir)
		image_ids = [x for x in image_ids if x[0]!="."]
		#sort image_ids
		image_ids.sort()
		# Add images
		for image_id in image_ids:
			if subset:
				self.add_image(
					"cell",
					image_id=image_id[0:image_id.find(".")],
					path=os.path.join(dataset_dir,subset))#,"images",image_id))#os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))
			else:
				self.add_image(
					"cell",
					image_id=image_id[0:image_id.find(".")],
					path=os.path.join(dataset_dir))

	def add_image(self, source, image_id, path, **kwargs):
		image_info = {
			"id": image_id,
			"source": source,
			"path": path,
		}
		image_info.update(kwargs)
		self.image_info.append(image_info)

	def prepare(self, class_map=None):
		"""Prepares the Dataset class for use.
		TODO: class map is not supported yet. When done, it should handle mapping
			  classes from different datasets to the same class ID.
		"""

		def clean_name(name):
			"""Returns a shorter version of object names for cleaner display."""
			return ",".join(name.split(",")[:1])

		# Build (or rebuild) everything else from the info dicts.
		self.num_classes = len(self.class_info)
		self.class_ids = np.arange(self.num_classes)
		self.class_names = [clean_name(c["name"]) for c in self.class_info]
		self.num_images = len(self.image_info)
		self._image_ids = np.arange(self.num_images)

		# Mapping from source class and image IDs to internal IDs
		self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
									  for info, id in zip(self.class_info, self.class_ids)}
		self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
									  for info, id in zip(self.image_info, self.image_ids)}

		# Map sources to class_ids they support
		self.sources = list(set([i['source'] for i in self.class_info]))
		self.source_class_ids = {}
		# Loop over datasets
		for source in self.sources:
			self.source_class_ids[source] = []
			# Find classes that belong to this dataset
			for i, info in enumerate(self.class_info):
				# Include BG class in all datasets
				if i == 0 or source == info['source']:
					self.source_class_ids[source].append(i)

	def load_mask(self, image_id):
		"""Generate instance masks for an image.
	   Returns:
		masks: A bool array of shape [height, width, instance count] with
			one mask per instance.
		class_ids: a 1D array of class IDs of the instance masks.
		"""
		# If not a balloon dataset image, delegate to parent class.
		image_info = self.image_info[image_id]
		if image_info["source"] != "cell":
			return super(self.__class__, self).load_mask(image_id)
			#see config.py for parent class default load_mask function

		# Get mask directory from image path
		mask_dir = os.path.join(image_info['path'], "gt")

		data = load_json_data(os.path.join(mask_dir,image_info['id']+".json")) #load file with same name.

		# Convert polygons to a bitmap mask of shape
		# [height, width, instance_count]
		mask = np.zeros([data["images"]["height"], data["images"]["width"], len(data['annotations']['regions']['area'])],
						dtype=np.uint8)
						#puts each mask into a different channel.
		for i,[verty,vertx] in enumerate(zip(data['annotations']['regions']['x_vert'],data['annotations']['regions']['y_vert'])):
			#alright, so this appears backwards (vertx, verty) but it is this way because of how matplotlib does plotting.
			#I have verified this notation is correct CE 11/20/20
			poly = np.transpose(np.array((vertx,verty)))
			rr, cc = skimage.draw.polygon(poly[:,0], poly[:,1], mask.shape[0:-1])
			RR, CC = skimage.draw.polygon_perimeter(poly[:,0], poly[:,1], mask.shape[0:-1])
			try:
				mask[rr,cc,i] = 1
			except:
				print("too many objects, needs debugging")
				print(self.image_info[image_id])
			#put each annotation in a different channel.

		return mask.astype(np.bool)

	def load_image(self, image_id, mask=None, avg_pixel=None):
		"""Load the specified image and return a [H,W,Z,1] Numpy array.
		"""
		#ultimately, we'll do enough convolutions to get that down to the correct size.
		image = skimage.io.imread(os.path.join(self.image_info[image_id]['path'],'images',self.image_info[image_id]['id']+'.ome.tif'))

		##making new aray and filling it is faster than using pad, but only if we use "zeros" and not "full".
		##for a nonzero padding value, it is slower this way.
		image = image.astype(np.float32)

		#sometimes images are loaded with range 0-1 rather than 0-255.
		if np.max(image)<=1.:
			image = image*255.0
			#again, we will return to a 0-1 range at the end.

		if avg_pixel is None:
			pad_val = scipy.stats.tmean(image.ravel(),(0,100)) #notice we are excluding the cell objects.
			image = image - pad_val
			image[image<0]=0 #clip values. #this clip values was at 1 before.
		else:
			image = image - avg_pixel
			image[image<0]=0

		#sometimes images load as H x W x Z, sometimes by Z x H x W. we need latter
		if len(image.shape)==2:
			image = np.expand_dims(image, axis = 0)

		if image.shape[2] < image.shape[0]:
			print("The shape of input is H x W x Z rather than Z x H x W")

		#roll axis.
		image = np.rollaxis(image, 0, 3)

		"""
		Removed padding at this step and placed in load_image_gt and load_image_inference
		"""
		if mask is not None:
			#load weight map
			bad_pixels = self.load_weight_map(image_id)
			mask = np.max(mask,axis=-1) #take max projection
			mask = np.expand_dims(mask,axis=-1) #add dimension for np.where
			bad_pixels=np.where(mask==True,False,bad_pixels) #anywhere an annotated object is, we don't want to cancel it out.
			#for each channel in image, set these to the mode of image.
			#determine the mean of small numbers.
			image = np.where(bad_pixels==True, 0.0, image)

		image = np.expand_dims(image, axis=-1) #add so Channel is "gray"
		#image output is shape=[H,W,Z,1]
		#the default for conv layers is channels last. i.e. input is [Batch_size, H, W, Z, CH]
		#image = image / np.max(image)
		#should currently be between the range of 0-255, conver to 0-1
		#Already float32 dtype.
		return image/255.

	def load_weight_map(self, image_id):
		"""Load unannotated regions so they do not contribute to loss
		"""
		# If not a balloon dataset image, delegate to parent class.
		image_info = self.image_info[image_id]
		# Get mask directory from image path
		try:
			mask_dir = os.path.join(image_info['path'], "gt")
			#os.path.join(os.path.dirname(os.path.dirname(image_info['path'])), "gt")

			data = load_json_data(os.path.join(mask_dir,image_info['id']+".json")) #load file with same name.
			# Convert polygons to a bitmap mask of shape
			# [height, width, instance_count]
			wmap = np.zeros([data["images"]["height"], data["images"]["width"],1],
							dtype=np.uint8)
							#puts each mask into a different channel.
			for verty,vertx in zip(data['pixelweight']['x_vert'],data['pixelweight']['y_vert']):
				#alright, so this appears backwards (vertx, verty) but it is this way because of how matplotlib does plotting.
				#I have verified this notation is correct CE 11/20/20
				poly = np.transpose(np.array((vertx,verty)))
				rr, cc = skimage.draw.polygon(poly[:,0], poly[:,1], wmap.shape[0:-1])
				wmap[rr,cc,0] = 1
				#put each annotation in a different channel.

			wmap = wmap.astype(np.bool)

		except:
			wmap = False #we dont' have shape yet. Still works with np.where.

		return wmap

	def run_prep(self):
		self.load_cell(self.dataset_dir)
		self.prepare()

	def load_image_gt(self, image_id):
		mask = self.load_mask(image_id)
		IM = self.load_image(image_id, mask=mask)
		return mask, IM

def load_json_data(pth):
	with open(pth) as f:
		data=json.load(f)
	return data
