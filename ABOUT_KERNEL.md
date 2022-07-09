# CUDA_LEARN
Study notes
Introduction to GPU Programming Summer School
========================

---

# Tutorial : Kernels

## What is a kernel?

* #### A function which operates on an element of data and is executed by a CUDA thread
* #### We launch enough threads such that the operation is performed on all elements
* #### Threads are mapped onto our data via a grid of thread blocks
* #### Each instance of the function (thread) must be able to identify its location in the grid 
* #### Kernels are launched/invoked by the host, but run on the GPU\*

(* Newer version of CUDA allow kernels to launch kernels, but numba doesn't support this yet)

---

## Kernel limitations

* #### Can only return data via arguments to the kernel function - no return value
* #### Kernels cannot perform input or output - no printing or reading/writing files
* #### Exception handling inside kernels is limited (no ```try```/```catch```)
* #### Only a subset of the host language (e.g. Python or C) is supported
* #### Threads execute in lockstep within *warps* of 32 threads which map onto *multiprocessors* (groups of CUDA cores, e.g. 8, 32 or 128, 192 etc depending on the architecture version)


The final limitation might seem unimportant, but this means that anything which causes one thread in a warp to wait (e.g. for data to arrive from memory) will cause all threads to wait. Similary branches, (if statements) are problematic. Each thread must execute both branches and then decide which result to keep. This may subvert traditional expectations of how to optimise code.

---

## Thread blocks/grids

* #### Threads within a block can make use of some *shared device memory* - more on that in tutorial 3
* #### All threads can read/write to global device memory. This is where all our device arrays have been so far
* #### There are hardware limitations on the number of threads per block
* #### The grid can be 1D, 2D or 3D

See https://numba.pydata.org/numba-doc/dev/cuda/cudapysupported.html for supported features. In particular note that numpy functions which dynamically create new arrays are not allowed in kernels.

## Trivial Example

This all makes more sense with an example. We'll use functionality with Numba to create kernels.

Let's start with the kernel we need to implement out own version of `cp.multiply` from CuPy in yesterday's example. Recall that we want to multiply each element in a 2D matrix by the corresponding element in another matrix. This suggests we use a 2D grid of threads with one thread per matrix element.

Our kernel looks like this:

```python
import numpy as np
from numba import cuda

@cuda.jit
def multiply_elements(a, b, c):
    """
    Element-wise multiplication of a and b stored in c.
    """

    # What elements of a,b and c should this thread operate on?
    tx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    ty = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    # Better make sure the indices tx adn ty are inside the array!
    if ty < a.shape[0] and tx < a.shape[1]:
        c[ty, tx] = a[ty, tx] * b[ty, tx]
```
We've *decorated* this function with the identifier ```@cuda.jit``` which requires some explanation. Numba contains functionality to turn python functions into complied GPU code when first invoked. This is known as "just In time compilation" and will be familiar to Julia fans. Note that numba can also "jit" functions which run on the host (CPU) which might be useful if wanting to make fair CPU vs GPU benchmarks.



The next part of the function uses variables defined for us by CUDA which give each thread (i.e. instance of the function) a unique element of the thread grid to operate on...


```python
cuda.threadIdx.x  # Index of this thread within its block (x - direction)
cuda.blockIdx.x   # Which block of threads is this (x - direction)
cuda.blockDim.x   # Number of threads in each block (x - direction)
```

... plus similar in the y direction (and z if in 3D). To get the global position inside the grid we perform the computation 

```python
tx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
```

i.e. add the thread index within the current block thread to the number of threads in all previous blocks.

Finally we check that the resulting indices `tx` and `ty` will have something to operate on by comparing them to corresponding array sizes before addressing the arrays to perform the desired computation **on this single element **. Note that I'm being sloppy here and assuming somebody else has made sure `a`, `b` and `c` are all the same size/shape.


```python
if ty < a.shape[0] and tx < a.shape[1]:
        c[ty,tx] = a[ty,tx] * b[ty,tx]
```

This may seem unnecessary but is essential good practice. We have to use a whole number of blocks in the grid, which can often mean we launch more threads than necessary. The extra threads shouldn't try to write to memory which lies outside of the array or they'd be overwriting other data which we might need!

Next we need to know how to launch the kernel. Start with a trivial example using a 3x3 matrix for both of the inputs arrays such that the output should just contain the squares of the input.

# Create an array for the input data and copy it to the device
```python
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float)  
d_a = cuda.to_device(a)
```
# Create an array for the output
```python
d_c = cuda.device_array((3, 3),dtype=np.float)
```
# Now launch one thread per element, passing the *device* arrays to the kernel function
```python
multiply_elements[(1, 1), (3, 3)](d_a, d_a, d_c)
```
The function call looks like a normal python function call, but the arguments are prefixed by a 2-element list which specifies the number of blocks in the grid (first element) and the threads per block (second argument). For our 2D array each element in the list is a tuple specifying the size in each grid direction.

Here I've used one block of 3x3 threads mapping onto my 3x3 matrices. Lets check the result.

# Copy data back from the device to the host
```python
c = d_c.copy_to_host()
print(c)
```
Hoorah!

I might equally well have used 3x3 blocks of 1 thread each, but for bigger problems there are some things to think about.

* Threads are organised into warps of 32 - block sizes which are not a multiple of 32 will end up wasting some of a multiprocessor
* It might be desirable to maximise block size and hence the number of threads with access to the same *shared memory* 
* Each device has a maximum number of threads allowed per block. This can be queried in CUDA C codes but for our purposes we can check [the wikipedia page on CUDA](https://en.wikipedia.org/wiki/CUDA).

## Revenge of the Porg

Let's return to our image example from tutorial 1 and roughly (i.e. inside the notebook) the time taken for the whole convolution.

First we need to re-read the image and create the 2D Gaussian we want to convolve with.
```python
from PIL import Image        # Import the Python Image Library
import cupy as cp            # NumPy-like interface to cuFFT etc
from timeit import default_timer as timer  # Timer

# Open an image file and convert to single colour (greyscale)
img = Image.open('porg.jpg').convert('L')
img_data = np.asarray(img,dtype=float)
dim = img_data.shape[0]
#dim = 250
#img_resized = img.resize((dim,dim))

# Define the Gaussian to volume with
width = 0.2
domain = np.linspace(-5, 5,dim)
gauss = np.exp(-0.5*domain**2/(width*width)) 
shift = int(dim/2)
gauss = np.roll(gauss,shift)
gauss2D = gauss[:,np.newaxis] * gauss[np.newaxis,:]
```
Create all the arrays we need and move the input data onto the device
```python
# Make the data complex
img_data_complex = img_data + 1j * np.zeros((dim,dim))
gauss2D_complex = gauss2D + 1j * np.zeros((dim,dim))

# Put the data on the device
img_data_complex_d = cp.asarray(img_data_complex)
gauss2D_complex_d  = cp.asarray(gauss2D_complex)
```
Perform the convolution the way we did it yesterday, using `cp.multiply`.
```python
t1 = timer()

# FFT the two input arrays
img_fft_d   = cp.fft.fft2(img_data_complex_d)
gauss_fft_d = cp.fft.fft2(gauss2D_complex_d)

# Multiply each element in fft_img by the corresponding image in fft_gaus
img_conv_d = cp.multiply(img_fft_d, gauss_fft_d) 

# Inverse Fourier transform
img_ifft_d = cp.fft.ifft2(img_conv_d)
        
# Copy result back to host
img_ifft = cp.asnumpy(img_ifft_d)

t2 = timer()

# Elapsed time (in milliseconds)
print("Convolution on GPU took : ",1000*(t2-t1)," milliseconds.")
```
Now let's do the convolution using our new kernel in place of `cp.multiply`.
```python
t1 = timer()  # Start timer

# FFT the two input arrays
img_fft_d   = cp.fft.fft2(img_data_complex_d)
gauss_fft_d = cp.fft.fft2(gauss2D_complex_d)

# Use the kernel to multiply on the device
threads_per_block = (32, 32)
blocks = dim // 32 + 1
blocks_per_grid = (blocks, blocks)

d_img_conv = cuda.device_array((dim, dim),dtype=np.complex)

multiply_elements[blocks_per_grid, threads_per_block](img_fft_d, gauss_fft_d, img_conv_d)

# Inverse Fourier transform
img_ifft_d = cp.fft.ifft2(img_conv_d)
        
# Copy result back to host
img_ifft = cp.asnumpy(img_ifft_d)

t2 = timer()

# Elapsed time (in milliseconds)
print("Convolution with multiplication on device took : ",1000*(t2-t1)," milliseconds.")
```
You might see that that is slower than using `cp.multiply`.

BUT - remember that our kernel is compiled on first use so this slower time include the time taken to compile the kernel. Run the above cell again and you should see that there's a significant improvement and that our kernel is providing pretting pretty much the same performance as the CuPy `cp.multiply`. 

Let's check the porg is OK and appropriately blurred.
```python
import matplotlib.pyplot as plt
%matplotlib inline

# Show the porg
plt.figure(figsize = [6, 6])
plt.imshow(img_ifft.real,cmap='gray');
```
## Exercises

* Cut and paste from the above to create a script which benchmarks the two methods of performing the convolution. Gather timings on the Tinis GPU node as an average over 10 runs. Remember to exclude the first run of the kernel-based method from your average if you want to measure only execution time.

* How much faster than the orignal CPU-based numpy implementation is this version?

* (Advanced) Can the CPU version be made faster by 'jitting' a function for the multiplication?


## Mandelbrot set

The kernel we've created above is about as trivial as it gets. They can be much more involved. As an example let's look at generation of the [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set). I'm borrowing very heavily here from an example workbook on GitHub, so full credit for this example goes to the original author.

https://github.com/harrism/numba_examples/blob/master/mandelbrot_numba.ipynb

As well as demonstrating that kernels can be less trivial, this also demonstrates the concept of a *device function*, i.e. a function which will only ever run on the GPU and will only be called from within a kernel. It cannot be called from the host.

The coordinates $x$, $y$ are part of the Mandelbrot set if the iterative map

$$ z_{i+1} = z_{i}^{2} + c $$

does not diverge when the complex numbers $z_{0} = 0$ and $ c = x + iy$. To make things graphical we colour pixels in the $x$, $y$ plane according to how rapidly this map diverges, i.e. how many iterations it takes the magnitude of $z$ to reach some threshold value.

The following function counts this number of iterations for a threshold value of 4 for input coordinates $x$ and $y$.
```python
#from numba import autojit
#@autojit
def mandel(x, y, max_iters):
  """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
  """
  c = complex(x, y)
  z = 0.0j
  for i in range(max_iters):
    z = z*z + c
    if (z.real*z.real + z.imag*z.imag) >= 4:
      return i

  return max_iters
```
As this stands `mandel` is a standard python function which will run on the CPU. We could copy and paste this into a new function `mandel_cpu` decorated with

```python
@cuda.jit(Device=True)
```

to indicate that it will be a device function. Numba provides an easier way to create device functions from standard python functions however.

# Create the device function mandel_gpu from the function "mandel" above
mandel_gpu = cuda.jit(device=True)(mandel)

Now all we need is a kernel which calls this function for each point in the $x$, $y$ plane. Some note on this...

* We calculate the pixel size on every thread, duplicating effort. The original author of this example may be assuming this is faster than calculating it once on the CPU and suffering an extra copy to device memory but you might want to experiment with that.


* Each instance of the kernel (thread) operates on a single element of the image array as in the porg example. For very large images this might not be practical. A more complicated kernel would operate on a chunk of pixels.
```python
@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
  
  # Get the dimensions of the grid from the image device array
  dimx = image.shape[1]
  dimy = image.shape[0]

  # Work out spacing between elements 
  pixel_size_x = (max_x - min_x) / dimx
  pixel_size_y = (max_y - min_y) / dimy

  # What elements of the image should this thread operate on?
  tx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
  ty = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

  # Coordinates in the complex plane
  real = min_x + tx * pixel_size_x
  imag = min_y + ty * pixel_size_y 
    
  # Count number of interations needed to diverge
  if ty < dimy and tx < dimx:
      image[ty, tx] = mandel_gpu(real, imag, iters)
```
We must create an image array, and specify what range we want this to represent along the real and imaginary axes.
```python
# Array to hold the output image - i.e. number of iterations 
# as an unsigned 8 bit integer
image = np.zeros((1000, 1500), dtype = np.uint8)

# Range over which we want to explore membership of the set
rmin = -2.0 ; rmax = 1.0
imin = -1.0 ; imax = 1.0

# Maximum number of iterations before deciding "does not diverge"
maxits = 20
```
First let's time how long it takes to populate the image on the CPU.
```python
t1 = timer() # Start timer

pixel_size_x = (rmax - rmin) / image.shape[1]
pixel_size_y = (imax - imin) / image.shape[0]

# This is probably the most non-pythonic way to do this...
for j in range(image.shape[0]):
    for i in range(image.shape[1]):
        
        real = rmin + i * pixel_size_x
        imag = imin + j * pixel_size_y
        
        image[j, i] = mandel(real, imag, maxits)
        
t2 = timer()

# Print time taken
print("Mandelbot created on CPU in : ",1000*(t2-t1)," milliseconds.")

# Display the image
#plt.figure(figsize = [9, 9])
#plt.imshow(image,cmap='RdBu',extent=[rmin, rmax, imin, imax]);
```
Now use our kernel to populate the image on the GPU. Note that I'm fixing the number of threads per block at 32 x 32, and then deciding how many blocks to launch from this and the size of the image.
```python
# The image size above is chosen to map onto a whole number of threadblocks. 
# IMPORTANT - we normally think of arrays indexed as row, column hence y, x
# The tuples specifiying the thread grid dimensions are indexed as x, y
threads_per_block = (32, 32) 

bx = image.shape[1] // threads_per_block[1] + 1
by = image.shape[0] // threads_per_block[0] + 1

blocks_per_grid = (bx, by)

t1 = timer() # Start timer

# Copy image to a device array which we will populate in our kernel
d_image = cuda.to_device(image)

# Launch the kernel, passing the range of x and y to use 
mandel_kernel[blocks_per_grid, threads_per_block](rmin, rmax, imin, imax, d_image, maxits) 

# Copy the resulting image back to the host
image = d_image.copy_to_host()

t2 = timer()  # Stop timer

print("Mandelbot created on GPU in : ",1000*(t2-t1)," milliseconds.")
```
Remember that you'll have to run this more than one to see how fast it executes once the kernel has already been compiled.
```python
# Display the image
plt.figure(figsize = [9, 9])
plt.imshow(image,cmap='RdBu',extent=[rmin, rmax, imin, imax]);
```
## Exercises

* The speedup obtained from using the GPU in this example is likely too good to be true (and is). By 'jitting' the CPU `mandel` function can you measure a more realistic speedup? HINT: see the comments where `mandel` is first defined.

* Cut and paste from the above to create a script which benchmarks the GPU vs CPU implemenation as a function of image resolution (size of the numpy array `image`). Run this on the Tinis GPU ndoes. How does speedup vary with problem size.

* (Advanced) Are the choices made for `blocks_per_grid` and `threads_per_block` optimal? You may need to generate rather large images to see any variation with these quantities. 
