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


__global__ void sobel_shared( unsigned char * g, unsigned char * s, std::size_t cols, std::size_t rows ) {

    extern __shared__ int shared_g[]; //mémoire partagée (local pour le bloc multi proc)

    auto outputBlockDim_x = blockDim.x-2;
    auto outputBlockDim_y = blockDim.y-2;

    auto gId_x = blockIdx.x * outputBlockDim_x + threadIdx.x;//id i global pour toute l'image
    auto gId_y = blockIdx.y * outputBlockDim_y  + threadIdx.y;//id j global pour toute l'image

    auto lId_x = threadIdx.x; //id i local au bloc pour la portion d'image du bloc 
    auto lId_y = threadIdx.y; //id j local au bloc pour la portion d'image du bloc

    if( gId_x < cols && gId_y < rows){
        shared_g[ lId_y * blockDim.x + lId_x ] = g[ gId_y * cols + gId_x ];//charger g dans la shared
    

        __syncthreads();//attendre que tous les threads aient chargé

        
        //traitement Sobol avec la mémoire partagée
        int h, v, res;
        // GREY = ( 307 * R + 604 * G + 113 * B ) / 1024

        if( lId_x > 0 && lId_x <= outputBlockDim_x && lId_y > 0 && lId_y <= outputBlockDim_y && gId_y < (rows-1) && gId_x < (cols-1)) {
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
        s [ gId_y * cols + gId_x ] = (int) sqrt((float)res);
        //s [ gId_y * cols + gId_x ] = 127;

        } 
    }
}


void print_if_err(cudaError_t erreur);

int main(int argc, char** argv)
{

    if(argc != 5){
        std::cout << "Error argument number" << std::endl << "Usage : <image_in> <image_out> <blockDim.x> <blockDim.y>, please verify that <image_in> is jpg and is in ./image/input/" << std::endl;
        exit(1);
      }
      
      const int blockSizeX = atoi(argv[3]);
      const int blockSizeY = atoi(argv[4]);
    
      if( (blockSizeX * blockSizeY) > 1024 ){
        std::cout << "Error block dimension" << std::endl << "<blockDim.x> * <blockDim.y> must be lower or equal to 1024. And both must be positive" << std::endl;
        exit(1);
      }
    
      std::string path_in = argv[1];
      path_in = "./images/input/" + path_in;
    
      std::string path_out = argv[2];
      path_out = "./images/output/" + path_out;


  cv::Mat m_in = cv::imread( path_in, cv::IMREAD_UNCHANGED );

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

  dim3 t( blockSizeX, blockSizeY ); //32, 32
  dim3 b( ( cols - 1) / t.x + 1 , ( rows - 1 ) / t.y + 1 );
  
  cudaEventCreate(&startK);
  cudaEventCreate(&stopK);
  cudaEventRecord(startK, 0);
  
  grayscale<<< b, t >>>( rgb_d, g_d, cols, rows );
  //cudaMemcpy( g.data(), g_d, rows * cols, cudaMemcpyDeviceToHost );
  cudaDeviceSynchronize();

  dim3 t_s( blockSizeX, blockSizeY ); //17, 32
  dim3 b_s( ( cols - 1) / (t_s.x-2) +1 , ( rows - 1 ) / (t_s.y-2) +1 );

  sobel_shared<<< b_s, t_s, (t_s.x)*(t_s.y)*sizeof(int)>>>(g_d, s_d, cols, rows);
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
  

  cv::imwrite( path_out, m_out );

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

