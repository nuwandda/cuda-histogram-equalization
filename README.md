# CUDA Histogram Equalization
Implementation of a histogram equalization program using CUDA. Histogram equalization is a technique for adjusting image intensities to enhance contrast.
## System Specifications
• Azure NC6 </br>
• Cores: 6 </br>
• GPU: Tesla K80 </br>
• Memory: 56 GB </br>
• Disk: 380 GB SSD </br>
The Tesla K80 delivers 4992 CUDA cores with a dual-GPU design, up to 2.91 Teraflops of double- precision and up to 8.93 Teraflops of single-precision performance.

## Implementation Details
  In this implementation, we have to main comparison topics. One is privatization and the other is scan type. We can run the implementation with privatization or without privatization. And also, we can run the implementation with Kogge-Stone or with Brent-Kung.
To run the code smoothly, please follow command line arguments given below.</br>
  •argv[1]: IMAGENAME </br>
  •argv[2]: PRIVATIZATION </br>
  •argv[3]: SCANTYPE </br>
  •argv[4]: OUTPUTIMAGENAME </br>
PRIVATIZATION represents the privatization type. To run the code without privatization, type ‘0’ for second argument. To run it with privatization, type ‘1’ for second argument.
SCANTYPE represents the scan type. To run the code with Kogge-Stone, type ‘0’ for third argument. To run it with Brent-Kung, type ‘1’ for the third argument.
An example command can be like below.</br>
</br>--> ./histogram 1.pgm 1 1 output.pgm </br>
</br>Kogge-Stone Adder is a parallel prefix adder that was developed by Peter M.Kogge and Harold S. Stone in 1973. Kogge-Stone is widely considered as a standard adder in the industry for high performance arithmetic circuits. The Dogge-Stone computes the carries in parallel and takes more area to implement, but has a lower fan-out at each stage, which then increases performance adder.
Brent-Kung Adder is a parallel prefix adder that was developed by Brent and Kung in 1982. The idea of Brent and Kung adder is to combine propagate signals and generate signals into groups of two by using the associative property only.
Brent-Kung Adder features with low network complexity comparing to Kogge-Stone Adder. The low network complexity assists to reduce the area of adder resulting in reducing the power consumption as well. This feature makes Brent-Kung more efficient than Kogge-Stone. On the other hand, Brent-Kung has more stages compare to Kogge-Stone. Having more competition stages leads to a slower adder. Hence, Kugge-Stone is more efficient than Brent-Kung in terms of speed. HISTOGRAM_SIZE is used as the block size of the kernel initialization, so I've had an equal number of threads and partition elements.
I have some variables that need to be explained. There are two CDF variables. cdf[] is for cdf calculations for all intensities and cdfClone[] is for equalize calculations. histogram[] array is for general histogram calculations.
  
  ## Implementation Performance
 ![Graph1](https://github.com/nuwandda/cuda-histogram-equalization/blob/main/mountain.jpg "Mountain Result") </br></br>
 ![Graph2](https://github.com/nuwandda/cuda-histogram-equalization/blob/main/apollonian.jpg "Apollonian Result") </br></br>
 ![Graph3](https://github.com/nuwandda/cuda-histogram-equalization/blob/main/brain.jpg "Brain Result") </br></br>
 ![Graph4](https://github.com/nuwandda/cuda-histogram-equalization/blob/main/dla.jpg "DLA Result") </br></br>
 ![Graph5](https://github.com/nuwandda/cuda-histogram-equalization/blob/main/x31.jpg "X31 Result") </br></br>
 As we can see in the graphs, there is no big differences between algorithms. The biggest differences are in Apollonian and mountain images. Both scan algorithms work good. There is no winner in scan algorithms but they have their best results in different images. However, they are almost the same.
For all images, privatization/shared memory is better than not privatization/shared memory one. Private histogram provides much less contention and serialization for accessing both private copies and the final copy.
For fractal images, times are higher than normal pictures. This may be because they have more different intensities for neighboring pixels.
