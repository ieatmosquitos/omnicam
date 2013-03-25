#include <cstdlib>
#include <iostream>
#include "cv.h"
#include "highgui.h"
#include "mutex.cpp"
#include "graphics.cpp"
#include <math.h>
#include <sstream>

#ifdef DEBUGMODE
#define DEBUGMESS(X) std::cout<<X<<std::endl
#else
#define DEBUGMESS(X)
#endif

// global variables
double * smooth_kernel;
unsigned int smooth_kernel_size;
double sigma;

double * edge_kernel;
unsigned int edge_kernel_rows;
unsigned int edge_kernel_cols;

class Fence:private Mutex{
  // private variables
  int _center[2];	// x and y position
  double _radius[2];	// radiuses of the two circles
  int * _frontier;	// <x,y,x',y'> TODO explain this
  unsigned int _frontier_resolution;	// the higher it is, the more scans are made
  double * _alpha;
  unsigned int _alpha_size;
  
  //private methods
  void _init();
  void _computeFrontier();
  void _computeAlpha();
  void _recomputeStuff();
  void _setResolution(double res);
  
public:
  Fence();
  Fence(int center_x, int center_y, double radius_inner, double radius_outer);
  int getCenter_X(){return _center[0];};
  int getCenter_Y(){return _center[1];};
  void setCenter(int x, int y);
  double getInnerRadius(){return _radius[0];};
  double getOuterRadius(){return _radius[1];};
  void paintFence(cv::Mat * image);
  void scaleInternalRadius(double amount);
  void scaleExternalRadius(double amount);
  cv::Mat * unrollImage(cv::Mat * image);
  cv::Mat * unrollImage2(cv::Mat * image);
};

struct callbackParams{
  cv::Mat * image;
  Fence * fence;
};

// useful methods

// MAX
template<class T> T max(T el1, T el2){
  if(el1 < el2) return el2;
  return el1;
}

// MIN
template<class T> T min(T el1, T el2){
  if(el1 < el2) return el1;
  return el2;
}

// STRINGIFY
template<class T> std::string stringify(T value) {
  std::ostringstream o;
  if (!(o << value))
    return "";
  return o.str();
}

// CHECKINSIDE
// checks whether a point is inside the image or not
template<class T> bool checkInside(T x, T y, cv::Mat * image){
  if(x >= image->cols) return false;
  if(y >= image->rows) return false;
  if(x < 0) return false;
  if(y < 0) return false;
  return true;
}


// Fence methods

void Fence::_init(){
  _alpha_size = _radius[1] - _radius[0];
  _setResolution(2 * M_PI * _radius[1]);
  this->_computeFrontier();
  this->_computeAlpha();
}

Fence::Fence(){
  this->_center[0] = 0;
  this->_center[1] = 0;
  this->_radius[0] = 10;
  this->_radius[1] = 30;
  
  this->_init();
}

Fence::Fence(int center_x, int center_y, double radius_inner, double radius_outer){
  this->_center[0] = center_x;
  this->_center[1] = center_y;
  this->_radius[0] = radius_inner;
  this->_radius[1] = radius_outer;
  
  this->_init();
}


void Fence::_computeFrontier(){
  DEBUGMESS("computing frontier...");
  _frontier = new int[4*_frontier_resolution];
  double current_angle = 0;
  for(unsigned int i=0; i<_frontier_resolution; i++){
    current_angle = (double)2 * M_PI * ((double)i/(double)_frontier_resolution);
    DEBUGMESS(std::string("current_angle: ").append(stringify(current_angle)));
    unsigned int index = i*4;
    double s = sin(current_angle);
    double c = cos(current_angle);
    DEBUGMESS("sin, cosine:");
    DEBUGMESS(stringify(s).append("\t").append(stringify(c)));
    _frontier[index] = _center[0] + c * _radius[0];
    _frontier[index+1] = _center[1] + s * _radius[0];
    _frontier[index+2] = _center[0] + c * _radius[1];
    _frontier[index+3] = _center[1] + s * _radius[1];
    DEBUGMESS("internal_x, internal_y, external_x, external_y");
    DEBUGMESS(stringify(_frontier[i]).append("\t").append(stringify(_frontier[i+1])).append("\t").append(stringify(_frontier[i+2])).append("\t").append(stringify(_frontier[i+3])));
  }
}


void Fence::_computeAlpha(){
  DEBUGMESS("computing alpha");
  _alpha_size = _radius[1] - _radius[0];
  DEBUGMESS(std::string("alpha size: ").append(stringify(_alpha_size)));
  _alpha = new double[_alpha_size];
  for (unsigned int i=0; i<_alpha_size; i++){
    _alpha[i] = 1 - ((double)i/(double)_alpha_size);
    DEBUGMESS(std::string("alpha[").append(stringify(i)).append("] = ").append(stringify(_alpha[i])));
  }
}


void Fence::_recomputeStuff(){
  DEBUGMESS("delete frontier");
  delete[] _frontier;
  DEBUGMESS("delete alpha");
  delete[] _alpha;
  _computeFrontier();
  _computeAlpha();
}


void Fence::setCenter(int x, int y){
  DEBUGMESS(std::string("set center: ").append(stringify(x)).append(" ").append(stringify(y)));
  this->lock();
  _center[0] = x;
  _center[1] = y;
  _recomputeStuff();
  this->unlock();
}


void Fence::_setResolution(double res){
  _frontier_resolution = (unsigned int) res;
  DEBUGMESS(std::string("frontier resolution: ").append(stringify(_frontier_resolution)));
}


void Fence::scaleInternalRadius(double amount){
  this->lock();
  double newsize = _radius[0] * amount;
  if(newsize>_radius[1]-1){
    this->unlock();
    return;
  }
  _radius[0] = newsize;
  _recomputeStuff();
  this->unlock();
}

void Fence::scaleExternalRadius(double amount){
  
  this->lock();
  double newsize = _radius[1] * amount;
  if(newsize < _radius[0]+1){
    this->unlock();
    return;
  }
  _radius[1] = newsize;
  _setResolution(2 * M_PI * _radius[1]);
  _recomputeStuff();
  this->unlock();
}


cv::Mat * Fence::unrollImage(cv::Mat * image){
  this->lock();
  cv::Mat * unrolled = new cv::Mat(_alpha_size, _frontier_resolution, CV_8UC3, cv::Scalar(0,0,0));
  int x;
  int y;
  for(unsigned int r=0; r<_alpha_size; r++){
    for(unsigned int c=0; c<_frontier_resolution; c++){
      unsigned int c_index = c*4;
      x = (int)(_alpha[r] * (double)(_frontier[c_index+2]) + (1-_alpha[r]) * (double)(_frontier[c_index]));
      y = (int)(_alpha[r] * (double)(_frontier[c_index+3]) + (1-_alpha[r]) * (double)(_frontier[c_index+1]));
      
      if(checkInside(x,y,image)){
	unrolled->at<cv::Vec3b>(r,c) = image->at<cv::Vec3b>(y,x);
      }
    }
  }
  
  this->unlock();
  return unrolled;
}


// another method for unrolling an image
// this should be slower.
cv::Mat * Fence::unrollImage2(cv::Mat * image){
  this->lock();
  cv::Mat * unrolled = new cv::Mat(_alpha_size, _frontier_resolution, CV_8UC3, cv::Scalar(0,0,0));
  unsigned int x;
  unsigned int y;
  
  double radius = _radius[1] - _radius[0];
  double angle;
  double cosangle;
  double sinangle;
  for(unsigned int c=0; c<_frontier_resolution; c++){
    angle = 2.0d * M_PI * ((double)c/(double)_frontier_resolution);
    cosangle = cos(angle);
    sinangle = sin(angle);
    
    for(unsigned int r = 0; r<_alpha_size; r++){
      x = (unsigned int)((double)(_center[0]) + _radius[0] * cosangle + _alpha[r] * radius * cosangle);
      y = (unsigned int)((double)(_center[1]) + _radius[0] * sinangle + _alpha[r] * radius * sinangle);
    
      if(checkInside(x,y,image)){
	unrolled->at<cv::Vec3b>(r,c) = image->at<cv::Vec3b>(y,x);
      }
    }
  }
  this->unlock();
  return unrolled;
}


// PAINTFENCE
// paints the fence on an image
void Fence::paintFence(cv::Mat * image){
  this->lock();
  
  int rows = image->rows;
  int cols = image->cols;
  
  cv::Vec3b red((uchar)0,(uchar)0,(uchar)255);
  
  DEBUGMESS("drawing central cross");
  // draw a cross in the center
  
  // if the center of the fence is inside the image
  if(checkInside(_center[0], _center[1], image)){
    
    // draw a cross
    for (int i = -rows/20; i<=rows/20; i++){
      if(checkInside(_center[0], _center[1]+i, image)){
	image->at<cv::Vec3b>(_center[1]+i, _center[0]) = red;
      }
    }
    for (int i = -rows/20; i<=rows/20; i++){
      if(checkInside(_center[0]+i, _center[1], image)){
	image->at<cv::Vec3b>(_center[1], _center[0]+i) = red;
      } 
    }
  }
  else{
    DEBUGMESS(std::string("center of the fence is outside the image: ").append(stringify(_center[0])).append(" ").append(stringify(_center[1])));
  }
  
  DEBUGMESS("drawing circles");
  // draw circles
  for(unsigned int i=0; i<_frontier_resolution; i++){
    unsigned int index = i*4;
    if(checkInside(_frontier[index], _frontier[index+1], image)){
      	image->at<cv::Vec3b>(_frontier[index+1], _frontier[index]) = red;
    }
    if(checkInside(_frontier[index+2], _frontier[index+3], image)){
      	image->at<cv::Vec3b>(_frontier[index+3], _frontier[index+2]) = red;
    }
  }
  
  DEBUGMESS("drawing lines");
  // draw lines at 0 and at pi/2 angles
  int x;
  int y;
  unsigned int pi_half_index = _frontier_resolution/4;
  pi_half_index = pi_half_index * 4;
  for(unsigned int i=0; i<_alpha_size; i++){
    x = (int)(_alpha[i] * (double)(_frontier[0]) + (1-_alpha[i]) * (double)(_frontier[2]));
    y = (int)(_alpha[i] * (double)(_frontier[1]) + (1-_alpha[i]) * (double)(_frontier[3]));
    if(checkInside(x,y,image)){
      image->at<cv::Vec3b>(y,x) = red;
    }
    
    x = (int)(_alpha[i] * (double)(_frontier[pi_half_index]) + (1-_alpha[i]) * (double)(_frontier[pi_half_index+2]));
    y = (int)(_alpha[i] * (double)(_frontier[pi_half_index+1]) + (1-_alpha[i]) * (double)(_frontier[pi_half_index+3]));
    if(checkInside(x,y,image)){
      image->at<cv::Vec3b>(y,x) = red;
    }
  }
  
  this->unlock();
}


// MOUSECALLBACK
void mouseCallBack(int event_type, int x, int y, int flags, void * param){
  switch( event_type ){
  	case CV_EVENT_LBUTTONDOWN:
	  {
	    
	    Fence * fence = ((callbackParams *)param)->fence;
	    fence->setCenter(x,y);
	    
	    break;
	  }
  	default:
	  break;
  }
  
  return;
}


// SETSMOOTHFILTER
// size MUST be odd
void setSmoothFilter(unsigned int size){
  // smooth_kernel_size = size;
  // unsigned int elements = size * size;
  
  // smooth_kernel = new double[elements];
  
  // for(unsigned int i=0; i<elements; i++){
  //   smooth_kernel[i] = (double)1/elements;
  // }
  
  int center = (size % 2);
  double total = 0;
  smooth_kernel_size = size;
  smooth_kernel = new double[size*size];
  for(int i=0; i<(int)size; i++){
    for(int j=0; j<(int)size; j++){
      double dx = abs(i-center);
      double dy = abs(j-center); 
      smooth_kernel[i*size + j] = exp(-(dx*dx + dy*dy) / (2.0 * sigma * sigma)) / sqrt(2.0 * M_PI * sigma * sigma);
      total += smooth_kernel[i*size + j];
    }
  }
  
  // normalize the weights
  // and print the new filter
  double check = 0;
  std::cout << "smoothing filter:" << std::endl;
  for(unsigned int i=0; i<smooth_kernel_size; i++){
    std::cout << "|";
    for(unsigned int j=0; j<smooth_kernel_size; j++){
      smooth_kernel[i*smooth_kernel_size + j] = (smooth_kernel[i*smooth_kernel_size + j])/total;
      std::cout << " " << smooth_kernel[i*smooth_kernel_size + j];
    }
    std::cout << " |" << std::endl;
  }
}


void init(){
  // set the smooth filter
  sigma = 2;
  setSmoothFilter(3);
  for(unsigned int i=0; i<smooth_kernel_size; i++){
    for(unsigned int j=0; j<smooth_kernel_size; j++){
      
    }
    std::cout << std::endl;
  }
  
  // set the edge filter
  edge_kernel_rows = 1;
  edge_kernel_cols = 3;
  edge_kernel = new double[3];
  for(unsigned int i=0; i<edge_kernel_cols * edge_kernel_rows; i = i+3){
    edge_kernel[i] = -1;
    edge_kernel[i+1] = 0;
    edge_kernel[i+2] = 1;
  }
}

// main method
int main(int argc, char ** argv){
  DEBUGMESS("DEBUGMODE ON");
  
  init();
  
  int capture_dev;
  if(argc < 2) capture_dev = CV_CAP_ANY;
  else capture_dev = atoi(argv[1]);
  
  CvCapture* capture = cvCaptureFromCAM(capture_dev);
  if(!capture){  
    fprintf(stderr, "ERROR: capture is NULL \n");  
    getchar();  
    return -1;  
  }
  
  
  // prepare for streaming
  const char * main_image_window_name = "streaming";
  const char * unrolled_window_name = "unrolled";
  const char * grayscaled_window_name = "grayscaled";
  const char * blurred_window_name = "blurred";
  const char * edges_window_name = "edges";
  const char * canny_window_name = "cannyfied";
  
  int frame_number = 0;
  IplImage * frame;
  cv::Mat * image;
  cv::Mat * unrolled;
  cv::Mat * blurred;
  cv::Mat * edges;
  cv::Mat * cannyfied;
  
  cvNamedWindow(main_image_window_name, CV_WINDOW_NORMAL);
  cv::moveWindow(main_image_window_name, 100, 100);
  cvNamedWindow(unrolled_window_name, CV_WINDOW_NORMAL);
  cv::moveWindow(unrolled_window_name, 500, 100);
  cvNamedWindow(grayscaled_window_name, CV_WINDOW_NORMAL);
  cv::moveWindow(grayscaled_window_name, 550, 150);
  cvNamedWindow(blurred_window_name, CV_WINDOW_NORMAL);
  cv::moveWindow(blurred_window_name, 600, 200);
  cvNamedWindow(edges_window_name, CV_WINDOW_NORMAL);
  cv::moveWindow(edges_window_name, 750, 100);
  cvNamedWindow(canny_window_name, CV_WINDOW_NORMAL);
  cv::moveWindow(canny_window_name, 800, 150);
  
  cvStartWindowThread;
  
  // first capture
  frame_number = cvGrabFrame(capture);
  
  frame = cvRetrieveFrame(capture,frame_number);
  if(!frame){
    fprintf(stderr,"ERROR: frame is null.. \n");
    getchar();
  }
  image = new cv::Mat(frame);
  
  // create the fence
  Fence fence(image->cols/2,image->rows/2,65,210);
  
  callbackParams params;
  params.image = image;
  params.fence = &fence;
  
  cvSetMouseCallback(main_image_window_name, mouseCallBack, &params);
  
  char key = -1;
  while(key!=27 && key!='q'){
    // check pressed button
    if(key!=-1){
      if(key=='o'){
	fence.scaleExternalRadius(1.1);
      }
      else if(key=='i'){
	fence.scaleExternalRadius(0.9);
      }
      else if(key=='l'){
	fence.scaleInternalRadius(1.1);
      }
      else if(key=='k'){
	fence.scaleInternalRadius(0.9);
      }
      else if(key=='s'){
	delete smooth_kernel;
	setSmoothFilter(smooth_kernel_size + 2);
      }
      else if(key=='a'){
	delete smooth_kernel;
	unsigned int newsize = max(1,(int)smooth_kernel_size - 2);
	setSmoothFilter(newsize);
      }
      else if(key=='x'){
	sigma = sigma + 0.2;
	std::cout << "new sigma: " << sigma << std::endl;
	delete smooth_kernel;
	setSmoothFilter(smooth_kernel_size);
      }
      else if(key=='z'){
	sigma = max(0.2, sigma - 0.2);
	std::cout << "new sigma: " << sigma << std::endl;
	delete smooth_kernel;
	setSmoothFilter(smooth_kernel_size);
      }
      else{
	std::cout << "key pressed: " << key << std::endl;
      }
    }
    
    DEBUGMESS("grabbing frame number");
    frame_number = cvGrabFrame(capture);
    
    DEBUGMESS("grabbing frame");
    frame = cvRetrieveFrame(capture,frame_number);
    
    if(!frame){
      fprintf(stderr,"ERROR: frame is null.. \n");
      getchar();
      continue;
    }
    
    DEBUGMESS("unrolling image");
    unrolled = fence.unrollImage(image);
    
    DEBUGMESS("getting grayscaled image");
    cv::Mat grayscaled;
    cv::cvtColor(*unrolled, grayscaled, CV_RGB2GRAY);
    
    DEBUGMESS("getting blurred image");
    blurred = applyFilter(&grayscaled, smooth_kernel, smooth_kernel_size);
    
    DEBUGMESS("applying derivative filter");
    edges = applyFilter(blurred, edge_kernel, edge_kernel_rows, edge_kernel_cols);
    
    DEBUGMESS("applying canny operator");
    cannyfied = canny(edges, 100,50);
    
    DEBUGMESS("drawing fence");
    fence.paintFence(image);
    
    DEBUGMESS("showing images");
    cv::imshow(main_image_window_name, *image);
    cv::imshow(unrolled_window_name, *unrolled);
    cv::imshow(grayscaled_window_name, grayscaled);
    cv::imshow(blurred_window_name, *blurred);
    cv::imshow(edges_window_name, *edges);
    cv::imshow(canny_window_name, *cannyfied);
    
    DEBUGMESS("deleting images");
    delete unrolled;
    delete blurred;
    delete edges;
    delete cannyfied;
    
    DEBUGMESS("read key...");
    key = (char) cv::waitKey(20); // non blocking, returns -1 if nothing was pressed
  }
    
  // cvReleaseCapture(&capture);
  cvDestroyWindow(main_image_window_name);
  
  delete image;
  
  return 0;
}
