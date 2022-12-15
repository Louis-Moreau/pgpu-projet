#include <opencv2/opencv.hpp>
#include <vector>


__global__ void grayscale_sobel_shared( unsigned char * rgb, unsigned char * s, std::size_t cols, std::size_t rows , std::size_t rowsOffset) {

    extern __shared__ int shared_g[];

    auto outputBlockDim_x = blockDim.x-2;
    auto outputBlockDim_y = blockDim.y-2;

    auto gId_x = blockIdx.x * outputBlockDim_x + threadIdx.x;//global id x
    auto gId_y = blockIdx.y * outputBlockDim_y + threadIdx.y;//global id y

    auto lId_x = threadIdx.x; //local id x
    auto lId_y = threadIdx.y; //local id y

    if( gId_x < cols && gId_y < rows){
      shared_g[ lId_y * blockDim.x + lId_x ] = (
			    307 * rgb[ 3 * ( (rowsOffset * cols) + gId_y * cols + gId_x ) ]
			    + 604 * rgb[ 3 * ( (rowsOffset * cols) + gId_y * cols + gId_x ) + 1 ]
			    + 113 * rgb[  3 * ( (rowsOffset * cols) + gId_y * cols + gId_x ) + 2 ]
			    ) >> 10;
        //shared_g[ lId_y * blockDim.y + lId_x ] = g[ gId_y * cols + gId_x ];//charger g dans la shared
            
        

        __syncthreads();//attendre que tous les threads aient chargé

        
        //traitement Sobel avec la mémoire partagée
        int h, v, res;
        // GREY = ( 307 * R + 604 * G + 113 * B ) / 1024

        if( lId_x > 0 && lId_x <= outputBlockDim_x && lId_y > 0 && lId_y <= outputBlockDim_y) {
        // Horizontal
            h =     shared_g[ ((lId_y - 1) * blockDim.x + lId_x - 1) ] -     shared_g[ ((lId_y - 1) * blockDim.x + lId_x + 1) ]
                + 2 * shared_g[ ( lId_y      * blockDim.x + lId_x - 1) ] - 2 * shared_g[ ( lId_y      * blockDim.x + lId_x + 1) ]
                +     shared_g[ ((lId_y + 1) * blockDim.x + lId_x - 1) ] -     shared_g[ ((lId_y + 1) * blockDim.x + lId_x + 1) ];

            // Vertical

            v =     shared_g[ ((lId_y - 1) * blockDim.x + lId_x - 1) ] -     shared_g[ ((lId_y + 1) * blockDim.x + lId_x - 1) ]
                + 2 * shared_g[ ((lId_y - 1) * blockDim.x + lId_x    ) ] - 2 * shared_g[ ((lId_y + 1) * blockDim.x + lId_x    ) ]
                +     shared_g[ ((lId_y - 1) * blockDim.x + lId_x + 1) ] -     shared_g[ ((lId_y + 1) * blockDim.x + lId_x + 1) ];

            h = min(h, 255);
            v = min(v, 255);
        
        res = h*h + v*v;
        //verif si > 255*255
        res = min(res, 255*255);

        // s[  ] = sqrt(res);
        s [ (rowsOffset * cols) + (gId_y * cols) + gId_x ] = (int) sqrt((float)res);
        //s [ gId_y * cols + gId_x ] = 127;

        } 
    }   
}


void print_if_err(cudaError_t erreur);

int main()
{
  cv::Mat m_in = cv::imread("in2.jpg", cv::IMREAD_UNCHANGED );

  auto rows = m_in.rows;
  auto cols = m_in.cols;

  unsigned char * rgb;
  unsigned char * s;
  unsigned char * rgb_d;
  unsigned char * s_d;

  const int batch = 1;
  int outChunkSize = (rows-1)/batch +1;
  int inChunkSize = outChunkSize + 2;

  cudaStream_t streams[batch];

  cudaMallocHost( &rgb, 3 * rows * cols * sizeof(char));
  cudaMallocHost( &s, rows * cols * sizeof(char));
  memcpy(rgb,m_in.data,3 * rows * cols * sizeof(char));

  cv::Mat m_out( rows, cols, CV_8UC1, s );

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  cudaMalloc( &rgb_d, 3 * rows * cols * sizeof(char));
  cudaMalloc( &s_d, rows * cols * sizeof(char));

  auto kernelOffset = 0;
  auto memCpyHtDOffset = 0;
  auto memCpyDtHOffset = 1;

  dim3 t_s( 32, 32 );
  dim3 b_s( ( cols - 1) / (t_s.x-2) +1 , (inChunkSize -1) / (t_s.y-2) +1 );
  cudaStreamCreate(&streams[0]);
  cudaMemcpyAsync(rgb_d + memCpyHtDOffset * cols * 3, rgb + memCpyHtDOffset * cols * 3,  3 * std::min(inChunkSize, rows - memCpyHtDOffset) * cols, cudaMemcpyHostToDevice, streams[0]); // this changes between here and for loop
  grayscale_sobel_shared<<< b_s, t_s, (t_s.x)*(t_s.y)*sizeof(int),streams[0]>>>(rgb_d, s_d, cols, std::min(inChunkSize, rows - kernelOffset),kernelOffset);
  cudaMemcpyAsync( s + memCpyDtHOffset * cols, s_d + memCpyDtHOffset * cols , std::min(outChunkSize, rows - memCpyDtHOffset) * cols, cudaMemcpyDeviceToHost, streams[0] );

  memCpyHtDOffset += inChunkSize; // this changes between here and for loop
  kernelOffset += outChunkSize;
  memCpyDtHOffset += outChunkSize;

  for (int i = 1; i < batch;i++) {
    std::cout << " y : " << kernelOffset << " - "<< std::min(inChunkSize, rows - kernelOffset) << std::endl;
    cudaStreamCreate( &streams[ i ]);
    cudaMemcpyAsync(rgb_d + memCpyHtDOffset * cols * 3, rgb + memCpyHtDOffset * cols * 3,  3 * std::min(outChunkSize, rows - memCpyHtDOffset) * cols, cudaMemcpyHostToDevice, streams[i]);
    grayscale_sobel_shared<<< b_s, t_s, (t_s.x)*(t_s.y)*sizeof(int),streams[i]>>>(rgb_d, s_d, cols, std::min(inChunkSize, rows - kernelOffset),kernelOffset);
    cudaMemcpyAsync( s + memCpyDtHOffset * cols, s_d + memCpyDtHOffset * cols , std::min(outChunkSize, rows - memCpyDtHOffset) * cols, cudaMemcpyDeviceToHost, streams[i] );

    memCpyHtDOffset += outChunkSize;
    kernelOffset += outChunkSize;
    memCpyDtHOffset += outChunkSize;
  }
  cudaError_t erreur_kernel = cudaGetLastError();
  print_if_err(erreur_kernel);

  cudaDeviceSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "All took " << elapsedTime << "ms" << std::endl;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cv::imwrite( "out4bis-cu.jpg", m_out );

  for (int i = 0; i < batch;i++) {
    cudaStreamDestroy( streams[ i ] );
  }

  cudaFree( rgb_d );
  cudaFree( s_d );

  return 0;
}

void print_if_err(cudaError_t erreur) {
    if(erreur != cudaSuccess) {
        std::cout<< "Erreur : " << cudaGetErrorString(erreur) <<  std::endl;
        exit(1);
    }
}

