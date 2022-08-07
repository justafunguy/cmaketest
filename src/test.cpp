#include<iostream>
#include<arm_neon.h>
using namespace std;
int main1()
{
    uint8_t *a=new uint8_t[8]{1,},*b=new uint8_t[8]{2,},*c=new uint8_t[8]{0,};
    
    uint8x8_t rega=vld1_u8(a);
    uint8x8_t regb=vld1_u8(b);
    uint8x8_t regc=vadd_u8(rega,regb);
    vst1_u8(c,regc);
    cout<<a[0]<<endl;
    return 0;
}