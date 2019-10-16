import numpy as np
from matplotlib import pyplot as plt, cm
from math import *
from scipy import ndimage as nd



def bounding_box(labeled_array,particle_id):
	particle = labeled_array==particle_id
	d = np.shape(np.shape(labeled_array))[0]
		
	if d == 2:
		x = np.sum(particle,axis=1)
		y = np.sum(particle,axis=0)

		x_nonzero = np.nonzero(x)
		y_nonzero = np.nonzero(y)
			

		xo = x_nonzero[0][0] ; xf = x_nonzero[0][-1]
		yo = y_nonzero[0][0] ; yf = y_nonzero[0][-1]
		
		particle_bbox = np.zeros((xf-xo+5,yf-yo+5))
		
		particle_bbox[1:-1,1:-1] = particle[xo-1:xf+2,yo-1:yf+2]
		
	if d == 3:	
		x =np.sum(np.sum(particle,axis=2),axis=1)
		y =np.sum(np.sum(particle,axis=2),axis=0)
		z =np.sum(np.sum(particle,axis=0),axis=0)
	
		x_nonzero = np.nonzero(x)
		y_nonzero = np.nonzero(y)
		z_nonzero = np.nonzero(z)

		xo = x_nonzero[0][0] ; xf = x_nonzero[0][-1]
		yo = y_nonzero[0][0] ; yf = y_nonzero[0][-1]
		zo = z_nonzero[0][0] ; zf = z_nonzero[0][-1]

		particle_bbox = np.zeros((xf-xo+3,yf-yo+3,zf-zo+3))
		
		particle_bbox[1:-1,1:-1,1:-1] = particle[xo:xf+1,yo:yf+1,zo:zf+1]
	
	
	return particle_bbox

def analysis(data,save=True,file_out='analysis.dat'):
	if file_out=='analysis.dat' and save:
		print('This is going to be saved on the folder containing')
		print('this script under the name : analysis.dat')
		print
		if raw_input('Continue ? (y/n) :\n').upper() <> 'Y':
			file_out = raw_input('Enter file output name : ')
		print


	# label the segmented image
	labeled_array,n = nd.label(data)

	# set the where the results will be saved
	data_out = np.zeros([n,17],dtype=np.float32)
	
	print('Starting analysis...')
	for i in range(1,n+1):
		print('Particle %s of %s.' % (i,n))

		# crop the volume containing the particle
		particle = bounding_box(labeled_array,i)

		# compute the centre of mass in the global frame of reference
		cm_global = nd.measurements.center_of_mass(labeled_array==i)

		# compute the centre of mass in the local frame of reference
		cm = nd.measurements.center_of_mass(particle)
		x0 = cm[0] ; y0 = cm[1] ; z0 = cm[2]

		# compute the volume of the particle
		vol = np.sum(np.sum(np.sum(particle,axis=0),axis=0),axis=0)

		# TENSOR OF INERTIA:
		# create the coordinates arrays
		lx, ly, lz = np.shape(particle)
		# span on each coordinate
		X=np.linspace(0,lx-1,lx)
		Y=np.linspace(0,ly-1,ly)
		Z=np.linspace(0,lz-1,lz)
		# create coordinates arrays
		x,y,z = np.meshgrid(X,Y,Z,indexing='ij')

		# compute the inertia arrays to be sum
		d_xx = ((y - y0)*particle)**2+((z - z0)*particle)**2
		d_yy = ((x - x0)*particle)**2+((z - z0)*particle)**2
		d_zz = ((x - x0)*particle)**2+((y - y0)*particle)**2
		d_xy = (x - x0)*(y - y0)*particle
		d_xz = (x - x0)*(z - z0)*particle
		d_yz = (y - y0)*(z - z0)*particle

		# compute the components of the tensor of inertia
		Ixx = np.sum(np.sum(np.sum(d_xx,axis=0),axis=0),axis=0)
		Iyy = np.sum(np.sum(np.sum(d_yy,axis=0),axis=0),axis=0)
		Izz = np.sum(np.sum(np.sum(d_zz,axis=0),axis=0),axis=0)
		Ixy = -np.sum(np.sum(np.sum(d_xy,axis=0),axis=0),axis=0)
		Ixz = -np.sum(np.sum(np.sum(d_xz,axis=0),axis=0),axis=0)
		Iyz = -np.sum(np.sum(np.sum(d_yz,axis=0),axis=0),axis=0)


		# rewrite the moment of inertia as a single line
		I = np.array([[Ixx,Ixy,Ixz],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]])
		
		# particle id
		data_out[i-1,0] = i
		# particle volume
		data_out[i-1,1] = vol
		# position of barycentre on the globa frame of reference
		data_out[i-1,2:5] = cm_global
		# components of the tensor of inertia
		data_out[i-1,5:14] = I.reshape(9)
		data_out[i-1,14:] = lx,ly,lz
		
	if save:
			header = '0-id \t1-vol \t2-x_cm \t3-y_cm \t4-z_cm \t05-Ixx \t06-Ixy \t07-Ixz \t08-Iyx \t09-Iyy \t10-Iyx \t11-Izx \t12-Izy \t13-Izz \t 14-lx \t 15-ly \t 16-lz'
			np.savetxt(file_out,data_out[::,:], delimiter="\t", fmt=["%i",]+["%5.2f",]*16,header=header)

	return data_out

