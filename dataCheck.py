import nibabel as nib
import os
import numpy as np


def discardBadFrames(n, ofStart: int = 2, ofEnd: int = 0):
    '''
    Removes unwanted frames from the beginning and end of the fMRI 
    '''
    dims = n.shape
    print('dims[-1]: ',dims[-1])
    return n[:,:,:,:,ofStart:(dims[-1]-ofEnd)]

def chunks(lst, n):
        """Yield successive n-sized chunks from the numpy array lst based on the last dimesions index."""
        dims = lst.shape
        temp = []
        for i in range(0, dims[-1]+1-n, n):
            temp.append(lst[:,:,:,i:i + n])
        return temp


def readAndSeparate(Filepath: str, chunksize: int = 6, ofStart: int = 2, ofEnd: int = 0):
    '''
    return : fingershrunk, footshrunk, lipsshrunk, nothingshrunk
    '''

    img = nib.load(Filepath)
    # turn the images into a 'data' format
    brainsdata = img.get_fdata()
    #print('img.shape: ', img.shape)

    # separate the first four entries as these correspond to images
    # before a movement starts from the bulk of images
    firstnothing = brainsdata[:,:,:,0:4]
    #print('nothing.shape', nothing.shape)
    brainsdata180 = brainsdata[:,:,:,4:]
    #print('brainsdata180: ', brainsdata180.shape)

    # grouping data into chunks of 4 images corresponding to 15 second intervals
    # and separate movement from 'nothing'(no movement) that
    # separates the movements
    brainsdatachunks = chunks(brainsdata180,chunksize)
    brainsdatchunks = np.asarray(brainsdatachunks)
    #print('brainsdatchunks: ', brainsdatchunks.shape)

    # grouping the chunks into arrays by type of movement

    finger = []
    foot = []
    lips = []
    nothinglst =[]
    dims = brainsdatchunks.shape
    #print('dims[0]: ',dims[0])


    for i in range(0,dims[0]):

        temp = brainsdatchunks[i,:,:,:]

        if i % chunksize == 0:
            finger.append(temp)
            #print('i: ', i.shape)
        if i % chunksize == 2:
            foot.append(temp)
        # print('i: ', i.shape)
        if i % chunksize == 4:
            lips.append(temp)
        # print('i: ', i.shape)
        else:
            nothinglst.append(temp)
        # print('i: ', i.shape)

    # converting to numpy arrays
    finger = np.asarray(finger)
    foot = np.asarray(foot)
    lips = np.asarray(lips)
    nothing = np.asarray(nothinglst)
    #print('nothing: ', nothing.shape)

    fingershrunk = discardBadFrames(finger, ofStart, ofEnd)
    footshrunk= discardBadFrames(foot, ofStart, ofEnd)
    lipsshrunk = discardBadFrames(lips, ofStart, ofEnd)
    nothingshrunk = discardBadFrames(nothing, ofStart, ofEnd)

    return fingershrunk, footshrunk, lipsshrunk, nothingshrunk

if __name__ == "__main__":
    # load fmri images for patient 5 finger foot lip movements
    Filepath = os.path.join('data', 'sub-05', 'ses-retest', 'func', 'sub-05_ses-retest_task-fingerfootlips_bold.nii.gz')
    fing, foot, lip, nothing = readAndSeparate(Filepath)
    print(fing.shape)
