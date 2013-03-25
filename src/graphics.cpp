#include "cv.h"
#include "highgui.h"
#include "vector"

struct Coordinate{
  int x;
  int y;
  
  Coordinate(int x, int y){
    this->x = x;
    this->y = y;
  }
};


const char * window = "padded";

template<class T> T abs(T value){
  if (value < 0) return -value;
  return value;
}


// APPLYFILTER
// applies a kernel to the image
// image has to be a grayscale image
// kernel is a vector of size kernel_size^2, representing a matrix with a side of kernel_size
// EG: this example puts the index number in each cell of a 3x3 matrix
//	| 0 1 2 |
//	| 3 4 5 |
//	| 6 7 8 |
// !!NOTE!! This will output the ABSOLUTE value of the computed pixels.
cv::Mat * applyFilter(cv::Mat * image, double * kernel, unsigned int kernel_size){
  
  cv::Mat * ret;
  
  if(kernel_size%2 == 0){	// kernel size must be odd
    ret = new cv::Mat(*image);
    return ret;
  }
  
  ret = new cv::Mat(image->rows, image->cols, CV_8UC1, cv::Scalar(0));
  
  unsigned int padding = kernel_size/2;
  cv::Mat padded(image->rows + 2*padding, image->cols + 2*padding , CV_8UC1);
  
  
  // create the padded image
  int x;
  int y;
  for(unsigned int r=0; r < padded.rows; r++){
    for(unsigned int c=0; c < padded.cols; c++){
      // set x and y
      x = c-padding;
      y = r-padding;
      
      // adjust x and y
      if(y < 0){
	y = 0;
      }
      else if(y >= image->rows){
	y = image->rows - 1;
      }
      
      if(x < 0){
	x = padding;
      }
      else if(c >= image->cols){
	x = image->cols - 1;
      }
      
      
      padded.at<uchar>(r,c) = image->at<uchar>(y,x);
    }
  }
  
  // generate the output image
  for(unsigned int r=0; r<ret->rows; r++){
    for(unsigned int c=0; c<ret->cols; c++){
      double value = 0;
      for(unsigned int kr=0; kr<kernel_size; kr++){
	for(unsigned int kc=0; kc<kernel_size; kc++){
	  value += kernel[kr*kernel_size + kc] * (double)(padded.at<uchar>(r+kr, c+kc));
	}
      }
      value = abs(value);
      ret->at<uchar>(r, c) = (unsigned int)value;
    }
  }
  
  return ret;
}


// APPLYFILTER revised for non-square kernels.
// the kernel size still has to be odd in both directions
// Kernel example:	| 0	1	2	3	4  |
//			| 5	6	7	8	9  |
//			| 10	11	12	13	14 |
cv::Mat * applyFilter(cv::Mat * image, double * kernel, unsigned int kernel_rows, unsigned int kernel_cols){
  
  cv::Mat * ret;
  
  if( (kernel_rows%2 == 0) || (kernel_cols%2 == 0) ){	// kernel size must be odd
    ret = new cv::Mat(*image);
    return ret;
  }
  
  ret = new cv::Mat(image->rows, image->cols, CV_8UC1, cv::Scalar(0));
  
  unsigned int padding_rows = kernel_rows/2;
  unsigned int padding_cols = kernel_cols/2;
  cv::Mat padded(image->rows + 2*padding_rows, image->cols + 2*padding_cols , CV_8UC1);
  
  
  // create the padded image
  int x;
  int y;
  for(unsigned int r=0; r < padded.rows; r++){
    for(unsigned int c=0; c < padded.cols; c++){
      // set x and y
      x = c-padding_cols;
      y = r-padding_rows;
      
      // adjust x and y
      if(y < 0){
	y = 0;
      }
      else if(y >= image->rows){
	y = image->rows - 1;
      }
      
      if(x < 0){
	x = padding_cols;
      }
      else if(c >= image->cols){
	x = image->cols - 1;
      }
      
      
      padded.at<uchar>(r,c) = image->at<uchar>(y,x);
    }
  }
  
  // generate the output image
  for(unsigned int r=0; r<ret->rows; r++){
    for(unsigned int c=0; c<ret->cols; c++){
      double value = 0;
      for(unsigned int kr=0; kr<kernel_rows; kr++){
	for(unsigned int kc=0; kc<kernel_cols; kc++){
	  value += kernel[kr*kernel_cols + kc] * (double)(padded.at<uchar>(r+kr, c+kc));
	}
      }
      value = abs(value);
      ret->at<uchar>(r, c) = (unsigned int)value;
    }
  }
  
  return ret;
}


void cannyExpandEdge(cv::Mat * image_in, cv::Mat * image_out, Coordinate position, unsigned int low_threshold, bool * visited){
  unsigned int cols = image_in->cols;
  unsigned int rows = image_in->rows;
  
  std::vector<Coordinate> checklist;
  checklist.push_back(position);
  
  unsigned int index;
  Coordinate * pixel;
  
  for(unsigned int i=0; i<checklist.size(); i++){
    pixel = &(checklist[i]);
    
    image_out->at<uchar>(pixel->y, pixel->x) = 255;
    
    // put neighbors in the to-check list
    for (int r=pixel->y - 1; r<=pixel->y + 1; r++){   // r will scan the rows
      //            std::cout<<"row:" << r << '\n';
      if((r<0)||(r>=rows)){
	//                std::cout<<"OUT OF BOUNDS -- continue\n";
	continue;
      }
      for (int c=pixel->x - 1; c<=pixel->x + 1; c++){       // c will scan the columns
	//                std::cout<<"col:" << c << '\n';
	if ((c<0)||(c>=cols)){  // don't consider out of bounds pixels
	  //                    std::cout<<"OUT OF BOUNDS -- continue\n";
	  continue;
	}
	
	if(!visited[r*cols+c]){
	  visited[r*cols+c] = true;
	  if(image_in->at<uchar>(r,c) > low_threshold){      // if it is colored
	    visited[(r*cols)+c] = true;	// set point as visited
	    checklist.push_back(Coordinate(c,r));
	  }
	}	
      }
    }
  }
  
}


cv::Mat * canny(cv::Mat * image, unsigned int up_threshold, unsigned int low_threshold){
  
  cv::Mat * ret = new cv::Mat(image->rows, image->cols, CV_8UC1, cv::Scalar(0));
  
  int total_cells = image->rows * image->cols;
  bool visited[total_cells];
  
  for(unsigned int i = 0; i<total_cells; i++){
    visited[i] = false;
  }
  
  for(unsigned int y=0; y<image->rows; y++){
    for(unsigned int x=0; x<image->cols; x++){
      if(!visited[(y*image->cols)+x]){
	if(image->at<uchar>(y,x) > up_threshold){
	  Coordinate pix(x,y);
	  cannyExpandEdge(image, ret, pix, low_threshold, visited);
	}
	else{
	  if(image->at<uchar>(y,x) < low_threshold){
	    visited[(y*image->cols)+x] = true;
	  }
	}
      }
    }
  }
  return ret;
}
  
// CANNYALL
// applies the whole canny operator: smooth + edge_detection + non_maximal_suppression(with hysteresis)
cv::Mat * cannyALL(cv::Mat * image, double * deriv_kernel, unsigned int deriv_kernel_size, double * smooth_kernel, unsigned int smooth_kernel_size, unsigned int up_threshold, unsigned int low_threshold){
  cv::Mat * blurred = applyFilter(image, smooth_kernel, smooth_kernel_size);
  
  cv::Mat * edges = applyFilter(blurred, deriv_kernel, deriv_kernel_size);
  
  delete blurred;
  
  cv::Mat * cannyfied = canny(edges, up_threshold, low_threshold);
  
  return cannyfied;
}
