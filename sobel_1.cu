#include <opencv2/opencv.hpp>
#include <vector>

__global__ void grayscale( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  // GREY = ( 307 * R + 604 * G + 113 * B ) / 1024
  if( i < cols && j < rows ) {
    g[ j * cols + i ] = (
			 307 * rgb[ 3 * ( j * cols + i ) ]
			 + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
			 + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
			 ) >> 10;
  }
}

__global__ void sobel( unsigned char * g, unsigned char * s, std::size_t cols, std::size_t rows ) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;
    int h, v, res;
    // GREY = ( 307 * R + 604 * G + 113 * B ) / 1024
    if( i < cols -1 && i > 0 && j < rows -1 && j > 0) {
       // Horizontal
	    h =     g[((j - 1) * cols + i - 1) ] -     g[((j - 1) * cols + i + 1) ]
            + 2 * g[( j      * cols + i - 1) ] - 2 * g[( j      * cols + i + 1) ]
            +     g[((j + 1) * cols + i - 1) ] -     g[((j + 1) * cols + i + 1) ];

        // Vertical

        v =     g[((j - 1) * cols + i - 1) ] -     g[((j + 1) * cols + i - 1) ]
            + 2 * g[((j - 1) * cols + i    ) ] - 2 * g[((j + 1) * cols + i    ) ]
            +     g[((j - 1) * cols + i + 1) ] -     g[((j + 1) * cols + i + 1) ];

        h = min(h, 255);
        v = min(v, 255);
       
       res = h*h + v*v;
       //verif si > 255*255
       res = min(res, 255*255);

       // s[  ] = sqrt(res);
       s [ j * cols + i ] = (int) sqrt((float)res);

    } 
}

void print_if_err(cudaError_t erreur);

int main()
{
  cv::Mat m_in = cv::imread("./images/input/in.jpg", cv::IMREAD_UNCHANGED );

  auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;
  
  std::vector< unsigned char > g( rows * cols );
  std::vector< unsigned char > s( rows * cols );
  
  cv::Mat m_out( rows, cols, CV_8UC1, s.data() );
  
  unsigned char * rgb_d;
  unsigned char * g_d;
  unsigned char * s_d;

  cudaEvent_t start,stop, startK, stopK;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  
  cudaMalloc( &rgb_d, 3 * rows * cols );
  cudaMalloc( &g_d, rows * cols );
  cudaMalloc( &s_d, rows * cols );

  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );

  dim3 t( 32, 32 );
  dim3 b( ( cols - 1) / t.x + 1 , ( rows - 1 ) / t.y + 1 );
  
  cudaEventCreate(&startK);
  cudaEventCreate(&stopK);
  cudaEventRecord(startK, 0);
  
  grayscale<<< b, t >>>( rgb_d, g_d, cols, rows );
  //cudaMemcpy( g.data(), g_d, rows * cols, cudaMemcpyDeviceToHost );
  cudaDeviceSynchronize();

  sobel<<< b, t >>>(g_d, s_d, cols, rows);
  cudaDeviceSynchronize();

  cudaEventRecord(stopK, 0);
  cudaEventSynchronize(stopK);
  float elapsedTimeK;
  cudaEventElapsedTime(&elapsedTimeK, startK, stopK);
  std::cout << "Kernel took " << elapsedTimeK << "ms" << std::endl;
  cudaEventDestroy(startK);
  cudaEventDestroy(stopK);




  cudaMemcpy( s.data(), s_d, rows * cols, cudaMemcpyDeviceToHost );

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "All took " << elapsedTime << "ms" << std::endl;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaError_t erreur_kernel = cudaGetLastError();
  print_if_err(erreur_kernel);
  

  cv::imwrite( "./images/output/out-cu.jpg", m_out );

  cudaFree( rgb_d );
  cudaFree( g_d );
  cudaFree( s_d );

  return 0;
}

void print_if_err(cudaError_t erreur) {
    if(erreur != cudaSuccess) {
        std::cout<< "Erreur : " << cudaGetErrorString(erreur) <<  std::endl;
        exit(1);
    }
}

