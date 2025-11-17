import pandas as pd
import scipy
import numpy as np


def main():
    mat = scipy.io.loadmat('umist_cropped.mat')
    facedat = mat['facedat']
    dirnames = mat['dirnames']

    print(mat.keys())
    print("Face data shape", facedat.shape)
    print("Dirnames shape", facedat.shape)
    print("People identifiers", mat['dirnames'][0])

    num_people = facedat.shape[1]
    print("Amount of different faces", num_people)

    all_images = []
    all_labels = []
    images_array = facedat[0]
    names_array = dirnames[0]

    for idx, subject in enumerate(images_array):
        images = subject

        # From (112,92,N) to (N, 112*92)
        N = images.shape[2]
        flat = images.reshape(-1, N).T

        all_images.append(flat)
        label = str(names_array[idx][0])
        all_labels.extend([label] * N)

    X = np.vstack(all_images)
    df = pd.DataFrame(X)
    df["Person"] = all_labels
    print(df.shape)
    print(df.head())


main()
