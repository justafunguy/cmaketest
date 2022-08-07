#include"neon_test.h"
#include<iostream>
using namespace std;
float32x4_t a={1,2,3,4},b={5,6,7,8};
void move()
{
    uint16x8_t a={0x1122,0x3344,0x5566,0x7788,0x99aa,0xbbcc,0xddee,0xff00};
    uint32x4_t ret=vmovl_high_u16(a);
    
    
    for(int i=0;i<4;++i)
    {
        cout<<hex<<ret[i]<<"\t";
    }
    cout<<endl;
   
}
void load()
{
    float16x8_t a={1,2,3,4,5,6,7,8};
    float16x4_t b=vdup_laneq_f16(a,2);
    for(int i=0;i<4;++i)
    {
        cout<<b[i]<<"\t";
    }
    cout<<endl;
    
}

void neon_test()
{
    //vst2q_u8(output + h * W + w, vld1q_u8_x2(input + index_in));
    uint8_t input1[8]={ 0,   2,   4,   6,   8,  10,12,14};
    uint8x8_t reg1=vld1_u8(input1);
    uint8_t input2[8]={1,   3,   5,   7,   9,  11,13,15};
    uint8x8_t reg2=vld1_u8(input2);
    uint8x16_t reg3=vcombine_u8(reg1,reg2);
    uint8_t output[16]={};
    vst2_u8(output,*((uint8x8x2_t*)(&reg3)));
    for(int i=0;i<16;++i)
        cout<<int(output[i])<<"\t";
    cout<<endl;
    
    
    
    
}


