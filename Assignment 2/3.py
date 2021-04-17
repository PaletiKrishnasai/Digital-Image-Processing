import numpy as np

x = np.random.randint(8,16,8)

def FFT(x):
    F=[]
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    for U in range(0,N):
        sum=0
        for y in range(0,N):
            sum=sum+(x[y]*(np.exp(-2j * np.pi * U * y / N)))
        F.append(sum)
    return(F)
x = np.random.random(10)
x = np.asarray(x)
print("\nUser-Defined\n",FFT(x))
print("\nInbuilt\n",np.fft.fft(x) )
print("\nCompare\n")
print(np.allclose(np.fft.fft(x),FFT(x)))