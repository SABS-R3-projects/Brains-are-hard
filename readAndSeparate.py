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
        i = 0

        while i < dims[-1]-1-n:
            temp.append(lst[:,:,:,i:i + n])
            i += n
        return temp



def readAndSeparate(Filepath: str, chunksize: int = 6, framesTillSignal: int = 4, ofStart: int = 2, ofEnd: int = 0):
    '''
    return : fingershrunk, footshrunk, lipsshrunk, nothingshrunk
    '''

    img = nib.load(Filepath)
    # turn the images into a 'data' format
    brainsdata = img.get_fdata()

    # separate the first four entries as these correspond to images
    # before a movement starts from the bulk of images
    firstnothing = brainsdata[:,:,:,0:framesTillSignal]
    brainsdata180 = brainsdata[:,:,:,framesTillSignal:]

    # grouping data into chunks of 4 images corresponding to 15 second intervals
    # and separate movement from 'nothing'(no movement) that
    # separates the movements
    brainsdatachunks = chunks(brainsdata180,chunksize)
    brainsdatchunks = np.asarray(brainsdatachunks)


    # grouping the chunks into arrays by type of movement

    finger = []
    foot = []
    lips = []
    nothinglst =[]
    dims = brainsdatchunks.shape

    for i in range(0,dims[0]):

        temp = brainsdatchunks[i,:,:,:]

        if i % chunksize == 0:
            finger.append(temp)
        if i % chunksize == 2:
            foot.append(temp)
        if i % chunksize == 4:
            lips.append(temp)
        else:
            nothinglst.append(temp)

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
    # need to take the 'nothing' that occurs after the movement away from each movement
    # to remove background 
    # fing[0] - nothing[0]
    # foot [0] - nothing [1]
    # lip [0] - nothing[2]
    # repeat
    fing, foot, lip, nothing = readAndSeparate(Filepath, framesTillSignal = 6)
    print(fing.shape)
