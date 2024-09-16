#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

typedef struct DATAREADER_ {
  uint32_t length;
  uint32_t w,h;
  FILE* images;
  FILE* labels;
  uint32_t index;
} datareader;

void swap_endian(uint32_t* i){
  uint32_t y=*i;
  *i=(y>>24)&0xff | (y>>8)&0xff00 | (y<<8)&0xff0000 | (y<<24)&0xff000000;
}

datareader* datareader_open(const char* imagename, const char* labelname){
  datareader* dr=(datareader*)malloc(sizeof(datareader));
  dr->images=fopen(imagename,"rb");  
  dr->labels=fopen(labelname,"rb");
  //read header
  uint32_t magic;
  fread(&magic,4,1,dr->images);
  swap_endian(&magic);
  fread(&dr->length,4,1,dr->images);
  swap_endian(&dr->length);
  fread(&dr->w,4,1,dr->images);
  swap_endian(&dr->w);
  fread(&dr->h,4,1,dr->images);
  swap_endian(&dr->h);
  fread(&magic,4,1,dr->labels);
  swap_endian(&magic);
  fprintf(stderr,"dataset of %d images of size %dx%d\n",dr->length,dr->w,dr->h);
  uint32_t lengthL;
  fread(&lengthL,4,1,dr->labels);
  swap_endian(&lengthL);
  //kill yourself if lengths are not the same
  assert(lengthL==dr->length);
  //initialize index
  dr->index=0;
  return dr;
}

void datareader_close(datareader* dr){
  fclose(dr->images);
  fclose(dr->labels);
  //prevent double close
  dr->images=0;
  dr->labels=0;
}

/* reads image into data pointer, label 0-9 into label, returns 1 on success, 0 on end of dataset */
int datareader_next(datareader* dr, uint8_t* data, uint8_t* label){
  // if we ran out of data
  if(dr->index == dr->length)return 0;
  
  fread(data,1,dr->w*dr->h,dr->images);
  fread(label,1,1,dr->labels);
  dr->index++;
  return 1;
}

int main(int argc, char** argz){
  datareader* dr = datareader_open("train-images-idx3-ubyte","train-labels-idx1-ubyte");
  uint8_t* image = (uint8_t*)malloc(dr->w*dr->h);
  uint8_t label;
  while(datareader_next(dr,image,&label)){
    //replace with real code
    printf("image:\n");
    for(int y=0;y<dr->h;y++){
      for(int x=0;x<dr->w;x++){
	printf("%d ",image[y*dr->w+x]);
      }
      printf("\n");
    }
    printf("label: %d\n",label);
    break;
  }
  datareader_close(dr);
  return 0;
}
