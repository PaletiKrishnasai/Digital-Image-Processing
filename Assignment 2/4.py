import numpy as np
s=8

x = np.random.random(s)


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

x = np.random.rand(s,s)
F2=[]
Ftemp=np.zeros((s,s),dtype=complex)
temp=[]

M = x.shape[0]


for i in range(0,M):
    temp.append(FFT(x[i]))
temp = np.array(temp)


N= x.shape[1]
for i in range(0,N):
    #print(FFT(temp[:,i]))
    Ftemp[:,i] = FFT(temp[:,i])

print("Built In\n")
print(np.fft.fft2(x))
print("\n")
print(Ftemp)
print("\n")
print("Compare\n")
print(np.allclose(np.fft.fft2(x),Ftemp))