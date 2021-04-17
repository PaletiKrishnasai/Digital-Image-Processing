import numpy as np
import cv2
import cmath
import matplotlib.pyplot as plt

lena = cv2.imread("lena.jpeg",0)
lena = np.array(lena)
dog = cv2.imread("dog.png",0)
dog = np.array(dog)

#Function for Fourier Transform
def FFT(x):
    F=[]
    #x = np.asarray(x, dtype=float)
    N = x.shape[0]
    for U in range(0,N):
        sum=0
        for y in range(0,N):
            sum=sum+(x[y]*(np.exp(-2j * np.pi * U * y / N)))
        F.append(sum)
    return(F)

row_FFT=[] #To store result of row-wise 1-dog FFT
col_FFT = np.zeros((lena.shape[0],lena.shape[1]),dtype=complex) #To store result of col-wise 1-dog FFT

M = lena.shape[0]
#row-wise FFT
for i in range(0,M):
    row_FFT.append(FFT(lena[i]))
row_FFT = np.array(row_FFT)

#col-wise FFT
N = lena.shape[1]
for i in range(0,N):
    col_FFT[:,i] = FFT(row_FFT[:,i])



#separataing phase and magnitude of Lena
lena_phase = [] 
lena_mag = []

lena_phase=np.angle(col_FFT)
lena_mag=np.log(np.abs(col_FFT))

lena_phase_inbuilt=np.angle(np.fft.fft2(lena))
lena_mag_inbuilt=np.log(np.abs(np.fft.fft2(lena)))

row_FFT=[]#To store result of row-wise 1-dog FFT
col_FFT = np.zeros((dog.shape[0],dog.shape[1]),dtype=complex) #To store result of col-wise 1-dog FFT

#row-wise  FFT dog
M = dog.shape[0]
print(dog.shape)
for i in range(0,M):
    row_FFT.append(FFT(dog[i]))

row_FFT = np.array(row_FFT)

#col-wise FFT dog
N = dog.shape[1]
for i in range(0,N):
    col_FFT[:,i] = FFT(row_FFT[:,i])

#separataing phase and magnitude of dog
dog_phase = [] 
dog_mag = []

dog_phase=np.angle(col_FFT)
dog_mag=np.log(np.abs(col_FFT))

dog_phase_inbuilt=np.angle(np.fft.fft2(dog))
dog_mag_inbuilt=np.log(np.abs(np.fft.fft2(dog)))

#plot Lena
plt.title('Lena phase')
plt.imshow(lena_phase,cmap='gray')
plt.show()
plt.title('Lena phase inbuilt')
plt.imshow(lena_phase_inbuilt,cmap='gray')
plt.show()
plt.title('Lena Mag')
plt.imshow(lena_mag,cmap='gray')
plt.show()
plt.title('Lena Mag inbuilt')
plt.imshow(lena_mag_inbuilt,cmap='gray')
plt.show()


#plot dog
plt.title('Dog phase')
plt.imshow(dog_phase,cmap='gray')
plt.show()
plt.title('Dog phase inbuilt')
plt.imshow(dog_phase_inbuilt,cmap='gray')
plt.show()
plt.title('Dog Mag')
plt.imshow(dog_mag,cmap='gray')
plt.show()

plt.title('Dog Mag inbuilt')
plt.imshow(dog_mag_inbuilt,cmap='gray')
plt.show()