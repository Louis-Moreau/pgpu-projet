#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>
#include <chrono>


int main()
{
  cv::Mat m_in = cv::imread("in2.jpg", cv::IMREAD_UNCHANGED );

  auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;

  std::vector< unsigned char > g( rows * cols );

  std::vector< unsigned char > s( rows * cols );

  cv::Mat m_out( rows, cols, CV_8UC1, s.data() );

  // Get starting timepoint
  auto start = std::chrono::high_resolution_clock::now();

  for( std::size_t i = 0 ; i < rows*cols ; ++i )
  {
    // GREY = ( 307 * R + 604 * G + 113 * B ) / 1024
    g[ i ] = ( 307 * rgb[ 3*i ]
		       + 604 * rgb[ 3*i+1 ]
		       + 113 * rgb[ 3*i+2 ]
		       ) >> 10;
  }
  
  unsigned int i, j, c;

  int h, v, res;


  for(j = 1 ; j < rows - 1 ; ++j) {

    for(i = 1 ; i < cols - 1 ; ++i) {

	// Horizontal
	h =     g[((j - 1) * cols + i - 1) ] -     g[((j - 1) * cols + i + 1) ]
	  + 2 * g[( j      * cols + i - 1) ] - 2 * g[( j      * cols + i + 1) ]
	  +     g[((j + 1) * cols + i - 1) ] -     g[((j + 1) * cols + i + 1) ];

	// Vertical

	v =     g[((j - 1) * cols + i - 1) ] -     g[((j + 1) * cols + i - 1) ]
	  + 2 * g[((j - 1) * cols + i    ) ] - 2 * g[((j + 1) * cols + i    ) ]
	  +     g[((j - 1) * cols + i + 1) ] -     g[((j + 1) * cols + i + 1) ];

	//h = h > 255 ? 255 : h;
	//v = v > 255 ? 255 : v;

	res = h*h + v*v;
	res = res > 255*255 ? res = 255*255 : res;

	s[ j * cols + i ] = sqrt(res);

    }

  }
  

  cv::imwrite( "./images/output/out_cpp.jpg", m_out );

  // Get ending timepoint
  auto stop = std::chrono::high_resolution_clock::now();
  // Get duration. Substart timepoints to
  // get duration. To cast it to proper unit
  // use duration cast method
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;
  
  return 0;
}
