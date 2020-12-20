##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
import os
import scipy.io as sio
from path import temp_database_path


##-----------------------------------------------------------------------------
##  Function
##-----------------------------------------------------------------------------
def createAccount(template, mask, name, exinfo):
	'''
	Description:
		Create an account in database based on extracted feature, and some
		extra information from the enroller.

	Input:
		template 	- Extracted template from the iris image
		mask		- Extracted mask from the iris image
		name		- Name of the enroller
		exinfo		- Extra information of the enroller
	'''
	# Get file name for the account
	files = []
	for file in os.listdir(temp_database_path):
	    if file.endswith(".mat"):
	        files.append(file)
	filename = str(len(files) + 1)

	# Save the file
	sio.savemat(temp_database_path + filename + '.mat',	\
		mdict={'template':template, 'mask':mask,\
		'name':name, 'exinfo':exinfo})

