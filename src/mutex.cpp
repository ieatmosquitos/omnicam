#include <pthread.h>
#include <iostream>

class Mutex{
  pthread_mutex_t _mutex;
  
 public:
  Mutex();
  ~Mutex();
  bool lock();
  bool unlock();
};

Mutex::Mutex(){
  while(pthread_mutex_init(&_mutex, NULL) == -1){
    usleep(2e4);
  }
}

Mutex::~Mutex(){
  while(pthread_mutex_destroy(&_mutex) == -1){
    usleep(2e4);
  }
}

bool Mutex::lock(){
  int mustbezero =  pthread_mutex_lock(&_mutex);
  if(mustbezero == 0){
    return true;
  }
  return false;
}

bool Mutex::unlock(){
  int mustbezero = pthread_mutex_unlock(&_mutex);
  if(mustbezero == 0){
    return true;
  }
  return false;
}
