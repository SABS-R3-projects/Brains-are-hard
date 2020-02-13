import nibabel as nib
import os
import numpy as np

# load fmri images for patient 5 finger foot lip movements
brains = os.path.join('data', 'sub-05', 'ses-retest', 'func', 'sub-05_ses-retest_task-fingerfootlips_bold.nii.gz')
img = nib.load(brains)
# turn the images into a 'data' format
brainsdata = img.get_fdata()
print('img.shape: ', img.shape)

# separate the first four entries as these correspond to images
# before a movement starts from the bulk of images
nothing = brainsdata[:,:,:,0:4]
print('nothing.shape', nothing.shape)
brainsdata180 = brainsdata[:,:,:,4:]
print('brainsdata180: ', brainsdata180.shape)

def chunks(lst, n):
    """Yield successive n-sized chunks from the numpy array lst based on the last dimesions index."""
    dims = lst.shape
    for i in range(0, dims[-1], n):
        yield lst[:,:,:,i:i + n]

# grouping data into chunks of 6 images corresponding to 30 second intervals
# TODO: probably change this to 3 and sperate movement from 'nothing'(no movement) that
#  separates the movements
brainsdatachunks = list(chunks(brainsdata180,6))

brainsdatchunks = np.asarray(brainsdatachunks)
# FIXME: This isn't the expected size probably somthing
print('brains: ', brainsdatchunks.shape)

# grouping the chunks into arrays by type of movement
# TODO: separate out no movement
# FIXME: These currently aren't the expected size

x = 0
finger = []
foot = []
lips = []

for i in brainsdatachunks:
    if x % 3 == 0:
        finger.append(i)
        #print('i: ', i.shape)
        x += 1
    if x % 3 == 1:
        foot.append(i)
       # print('i: ', i.shape)
        x += 1
    if x % 3 == 2:
        lips.append(i)
       # print('i: ', i.shape)
        x += 1

finger = np.asarray(finger)
print('finger: ', finger.shape)
