import numpy as np
import cv2
import cmath
import matplotlib.pyplot as plt

lena = cv2.imread("lena1.jpeg",0)
lena = np.array(lena)
dog = cv2.imread("dog.png",0)
dog = np.array(dog)

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

def IFFT(x):
    F=[]
    #x = np.asarray(x, dtype=float)
    N = x.shape[0]
    for U in range(0,N):
        sum=0
        for y in range(0,N):
            sum=sum+(x[y]*(np.exp(2j * np.pi * U * y / N)))
        F.append(sum/N)
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


#combining phase and magnitude of dog and lena resp.  
combined=np.multiply(lena_mag,np.exp(1j*dog_phase))


#IFFT to get output
M = combined.shape[0]
row_IFFT=[]
col_IFFT = np.zeros((combined.shape[0],combined.shape[1]),dtype=complex)

#performing row-wise 1-d IFFT
for i in range(0,M):
    row_IFFT.append(IFFT(combined[i]))
row_IFFT = np.array(row_IFFT)
print(row_IFFT.shape)
N = combined.shape[1]
for i in range(0,N):
    col_IFFT[:,i] = IFFT(row_IFFT[:,i])
IFFTcombined=np.asarray(col_IFFT)


plt.title('Lena Mag + Dog Phase')
plt.imshow( np.real(IFFTcombined),cmap='gray')
plt.show()
plt.title('Lena mag + Dog phase inbuilt')
plt.imshow(np.real(np.fft.ifft2(combined)),cmap='gray')
plt.show()