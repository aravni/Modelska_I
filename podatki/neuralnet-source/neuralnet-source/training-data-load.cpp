#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <assert.h>

class datareader {
private:
  FILE* images;
  FILE* labels;
  void swap_endian(uint32_t& i){
    i=(i>>24)&0xff | (i>>8)&0xff00 | (i<<8)&0xff0000 | (i<<24)&0xff000000;
  }
public:
  uint32_t length;
  uint32_t w,h;
  uint32_t index;

  datareader(const char* imagename, const char* labelname){
    images=fopen(imagename,"rb");  
    labels=fopen(labelname,"rb");
    //read header
    uint32_t magic;
    fread(&magic,4,1,images);
    swap_endian(magic);
    fread(&length,4,1,images);
    swap_endian(length);
    fread(&w,4,1,images);
    swap_endian(w);
    fread(&h,4,1,images);
    swap_endian(h);
    fread(&magic,4,1,labels);
    swap_endian(magic);
    fprintf(stderr,"dataset of %d images of size %dx%d\n",length,w,h);
    uint32_t lengthL;
    fread(&lengthL,4,1,labels);
    swap_endian(lengthL);
    //will raise alarms if it's different
    assert(lengthL==length);
    //initialize index
    index=0;
  }
  //destructor (cleanup)
  ~datareader(){
    fclose(images);
    fclose(labels);
  }
  /* reads image into data pointer, label 0-9 into label, returns 1 on success, 0 on end of dataset */
  int next(uint8_t* data, uint8_t* label){
    // if we ran out of data
    if(index == length)return 0;
    
    fread(data,1,w*h,images);
    fread(label,1,1,labels);
    index++;
    return 1;
  }
  
};

int main(int argc, char** argz){
  datareader dr("train-images-idx3-ubyte","train-labels-idx1-ubyte");
  uint8_t* image = new uint8_t[dr.w*dr.h];
  uint8_t label;
  while(dr.next(image,&label)){
    //replace with real code
    printf("image:\n");
    for(int y=0;y<dr.h;y++){
      for(int x=0;x<dr.w;x++){
	printf("%d ",image[y*dr.w+x]);
      }
      printf("\n");
    }
    printf("label: %d\n",label);
    break;
  }
  return 0;
}
