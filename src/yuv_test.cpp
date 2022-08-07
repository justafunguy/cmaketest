#include <iostream>
#include<stdio.h>
#include<stdlib.h>
#include<cstring>
#include<arm_neon.h>
using namespace std;
static bool pixel_shuffle(unsigned char *input, unsigned char *output, int index_in, int h_start, int h_end, int W, int pitch, int factor)
{
    if (!input || !output || h_start < 0 || (h_end <= h_start) || W <= 0 || factor <= 0 || (h_end - h_start) % factor != 0 || W % factor != 0)
        return false;
    if (factor != 1)
    {
        for (int base_h = h_start; base_h < factor + h_start; ++base_h)
        {
            for (int base_w = 0; base_w < factor; ++base_w)
            {
                for (int h = base_h; h < h_end; h += factor)
                {
                    for (int w = base_w; w < W; w += factor)
                    {
                        output[h * W + w] = input[index_in++];
                    }
                    index_in += pitch - W / factor % pitch;
                }
            }
        }
    }
    else
    {
        for (int h = h_start; h < h_end; ++h)
        {
            for (int w = 0; w < W / 2; ++w)
            {
                output[h * W + w] = input[index_in++];
            }
            index_in += pitch - (W / 2 % pitch);
            for (int w = W / 2; w < W; ++w)
            {
                output[h * W + w] = input[index_in++];
            }
            index_in += pitch - (W / 2 % pitch);
        }
    }
    return true;
}
static bool neon_pixel_shuffle(unsigned char *input, unsigned char *output, int index_in, int h_start, int h_end,int H, int W, int pitch, int factor,int channel)
{
    if (!input || !output || h_start < 0 || (h_end <= h_start) || W <= 0 || factor <= 0 || (h_end - h_start) % factor != 0 || W % factor != 0)
        return false;
    if (factor != 1)
    {
        int yuv_width=W/factor;
        int yuv_height=H*factor/channel;
        printf("yuv_height:%d   yuv_width:%d\n",yuv_height,yuv_width);
        if(factor==2)
        {
            int alignedchannelsize=yuv_height*pitch;
            int nr_step=yuv_width/16;
            int output_h=0,output_w=0;
            for(int f=0;f<factor;++f)
            {
                unsigned char*p_channel_1=input+index_in+2*f*alignedchannelsize;
                unsigned char*p_channel_2=p_channel_1+alignedchannelsize;
                for(int h=0, output_h=f;h<yuv_height;++h,output_h+=factor)
                {
                    for(int i=0,output_w=0;i<nr_step;++i,output_w+=32)
                    {
                        uint8x16_t temp1=vld1q_u8(p_channel_1);
                        uint8x16_t temp2=vld1q_u8(p_channel_2);
                        uint8_t temp[32];
                        vst1q_u8(temp,temp1);
                        vst1q_u8(temp+16,temp2);
                        vst2q_u8(output+output_h*W+output_w,*(uint8x16x2_t*)(&temp));
                        p_channel_1+=16;
                        p_channel_2+=16;
                    }
                    for(;output_w<W;output_w+=2)
                    {
                        output[output_h*W+output_w]=*p_channel_1;
                        ++p_channel_1;
                        output[output_h*W+output_w+1]=*p_channel_2;
                        ++p_channel_2;
                    }
                    p_channel_1+=pitch-yuv_width;
                    p_channel_2+=pitch-yuv_width;
                    
                }
            }
        }
    }
    else
    {
        for (int h = h_start; h < h_end; ++h)
        {
            int hW = W / 2;
            for (int k = 0; k < 2; ++k)
            {
                int temp = hW * (k + 1);
                int w = hW * k;
                int nr_step = hW / 64;
                while (nr_step--)
                {
                    vst1q_u8_x4(output + h * W + w, vld1q_u8_x4(input + index_in));
                    index_in += 64;
                    w += 64;
                }
                for (; w < temp; ++w)
                {
                    output[h * W + w] = input[index_in++];
                }
                index_in += pitch - (W / 2 % pitch);
            }
        }
    }
    return true;
}
bool neon_pixel_unshuffle(unsigned char *input, unsigned char *output, int index_out, int h_start, int h_end, int W, int pitch, int factor)
{
    if (!input || !output || h_start < 0 || (h_end <= h_start) || W <= 0 || factor <= 0 || (h_end - h_start) % factor != 0 || W % factor != 0)
        return false;
    if (factor != 1)
    {
        for (int base_h = h_start; base_h < factor + h_start; ++base_h)
        {
            for (int base_w = 0; base_w < factor; ++base_w)
            {
                for (int h = base_h; h < h_end; h += factor)
                {
                    int w = base_w;
                    if (factor == 2)
                    {
                        int nr_step = (W - w) / 32;
                        for (int t = 0; t < nr_step; ++t)
                        {
                            vst1q_u8(output + index_out, (vld2q_u8(input + h * W + w)).val[0]);
                            index_out += 16;
                            w += 32;
                        }
                        if (W - w >= 16)
                        {
                            vst1_u8(output + index_out, (vld2_u8(input + h * W + w)).val[0]);
                            index_out += 8;
                            w += 16;
                        }
                    }
                    if (factor == 4)
                    {
                        int nr_step = (W - w) / 64;
                        for (int k = 0; k < nr_step; ++k)
                        {
                            vst1q_u8(output + index_out, (vld4q_u8(input + h * W + w)).val[0]);
                            index_out += 16;
                            w += 64;
                        }
                        if (W - w >= 32)
                        {
                            vst1_u8(output + index_out, (vld4_u8(input + h * W + w)).val[0]);
                            index_out += 8;
                            w += 32;
                        }
                    }
                    for (; w < W; w += factor)
                        output[index_out++] = input[h * W + w];
                    index_out += pitch - W / factor % pitch;
                }
            }
        }
    }
    else
    {
        for (int h = h_start; h < h_end; ++h)
        {
            int hW = W / 2;
            for (int k = 0; k < 2; ++k)
            {
                int temp = hW * (k + 1);
                int w = hW * k;
                int nr_step = hW / 64;
                cout<<"nr_step:"<<nr_step<<endl;
                for(int i=0;i<nr_step;++i)
                {
                    vst1q_u8_x4(output + index_out, vld1q_u8_x4(input + h * W + w));
                    index_out += 64;
                    w += 64;
                }
                if (temp - w >= 48)
                {
                    vst1q_u8_x3(output + index_out, vld1q_u8_x3(input + h * W + w));
                    index_out += 48;
                    w += 48;
                }
                if (temp - w >= 32)
                {
                    vst1q_u8_x2(output + index_out, vld1q_u8_x2(input + h * W + w));
                    index_out += 32;
                    w += 32;
                }
                if (temp - w >= 24)
                {
                    vst1_u8_x3(output + index_out, vld1_u8_x3(input + h * W + w));
                    index_out += 24;
                    w += 24;
                }
                if (temp - w >= 16)
                {
                    vst1_u8_x3(output + index_out, vld1_u8_x3(input + h * W + w));
                    index_out += 16;
                    w += 16;
                }
                if (temp - w >= 8)
                {
                    vst1_u8(output + index_out, vld1_u8(input + h * W + w));
                    index_out += 8;
                    w += 8;
                }
                for (; w < temp; ++w)
                {
                    output[index_out++] = input[h * W + w];
                }
                index_out += pitch - (W / 2 % pitch);
            }
        }
    }
    return true;
}
static bool pixel_unshuffle(unsigned char *input, unsigned char *output, int index_out, int h_start, int h_end, int W, int pitch, int factor)
{
    if (!input || !output || h_start < 0 || (h_end <= h_start) || W <= 0 || factor <= 0 || (h_end - h_start) % factor != 0 || W % factor != 0)
        return false;
    if (factor != 1)
    {
        for (int base_h = h_start; base_h < factor + h_start; ++base_h)
        {
            for (int base_w = 0; base_w < factor; ++base_w)
            {
                for (int h = base_h; h < h_end; h += factor)
                {
                    for (int w = base_w; w < W; w += factor)
                    {
                        output[index_out++] = input[h * W + w];
                    }
                    index_out += pitch - W / factor % pitch;
                }
            }
        }
    }
    else
    {
        for (int h = h_start; h < h_end; ++h)
        {
            for (int w = 0; w < W / 2; ++w)
            {
                output[index_out++] = input[h * W + w];
            }
            index_out += pitch - (W / 2 % pitch);
            for (int w = W / 2; w < W; ++w)
            {
                output[index_out++] = input[h * W + w];
            }
            index_out += pitch - (W / 2 % pitch);
        }
    }
    return true;
}

int YUV_PostProcess(unsigned char *input, unsigned char *output, int H, int W, int pitch, int factor, bool yonly, bool use_neon)
{
    if (!input || !output)
    {
        return -1;
    }
    int Y_height = H * 2 / 3;
    int UV_height = Y_height / 2;
    int half_UV_height = UV_height / 2;

    if (use_neon)
    {
        if (!neon_pixel_shuffle(input, output, 0, 0, Y_height,H, W, pitch, factor,6))
            return -1;

        if (yonly)
            return 0;
        int index = factor * factor * pitch * Y_height / factor;
        if (!neon_pixel_shuffle(input, output, index, Y_height, Y_height + half_UV_height,H, W, pitch, factor / 2,6))
            return -1;
        if (!neon_pixel_shuffle(input, output, index * 1.25, Y_height + half_UV_height, H,H, W, pitch, factor / 2,6))
            return -1;
        return 0;
    }
    else
    {
        if (!pixel_shuffle(input, output, 0, 0, Y_height, W, pitch, factor))
            return -1;

        if (yonly)
            return 0;
        int index = factor * factor * pitch * Y_height / factor;
        if (!pixel_shuffle(input, output, index, Y_height, Y_height + half_UV_height, W, pitch, factor / 2))
            return -1;
        if (!pixel_shuffle(input, output, index * 1.25, Y_height + half_UV_height, H, W, pitch, factor / 2))
            return -1;
        return 0;
    }
}
int YUV_PreProcess(unsigned char *input, unsigned char *output, int H, int W, int pitch, int factor, bool yonly, bool use_neon)
{
    if (!input || !output)
    {
        return -1;
    }
    int Y_height = H * 2 / 3;
    int UV_height = Y_height / 2;
    int half_UV_height = UV_height / 2;
    if (use_neon)
    {
        if (!neon_pixel_unshuffle(input, output, 0, 0, Y_height, W, pitch, factor))
            return -1;
        if (yonly)
            return 0;
        int index = factor * factor * pitch * Y_height / factor;
        if (!neon_pixel_unshuffle(input, output, index, Y_height, Y_height + half_UV_height, W, pitch, factor / 2))
            return -1;
        if (!neon_pixel_unshuffle(input, output, index * 1.25, Y_height + half_UV_height, H, W, pitch, factor / 2))
            return -1;
        return 0;
    }
    else
    {
        if (!pixel_unshuffle(input, output, 0, 0, Y_height, W, pitch, factor))
            return -1;
        if (yonly)
            return 0;
        int index = factor * factor * pitch * Y_height / factor;
        if (!pixel_unshuffle(input, output, index, Y_height, Y_height + half_UV_height, W, pitch, factor / 2))
            return -1;
        if (!pixel_unshuffle(input, output, index * 1.25, Y_height + half_UV_height, H, W, pitch, factor / 2))
            return -1;
        return 0;
    }
}
#include<stdio.h>
#include<iostream>
using namespace std;
void yuv_test1()
{
    int size1=1*6*40;
    int size2=6*2*(20+5);
    unsigned char*yuv=(unsigned char*)malloc(size1);
    unsigned char *packed = (unsigned char *)malloc(size2);
    unsigned char *packed_neon = (unsigned char *)malloc(size2);
    unsigned char *unpacked = (unsigned char *)malloc(size1);
    unsigned char *unpacked_neon = (unsigned char *)malloc(size1);

    if(!yuv || !packed || !unpacked || !unpacked_neon)
    {
        printf("malloc failed!");
        return ;
    }
    for(int i=0;i<size1;++i)
    {
        yuv[i]=i;
    }
    
    YUV_PreProcess(yuv,packed,6,40,25,2,false,false);
    YUV_PreProcess(yuv,packed_neon,6,40,25,2,false,true);
    
    YUV_PostProcess(packed,unpacked,6,40,25,2,false,false);
    YUV_PostProcess(packed_neon,unpacked_neon,6,40,25,2,false,true);
    for(int i=0;i<size2;++i)
    {
        if(packed[i]!=packed_neon[i])
        {
            printf("preprocess error! \n");
            cout<<int(packed[i])<<" "<<int(packed_neon[i])<<endl;
            break;
        }
    }
    for(int i=0;i<size1;++i)
    {
        if(unpacked[i]!=unpacked_neon[i])
        {
            printf("postprocess failed!\n");
            break;
        }
    }
    for(int i=0;i<6;++i)
    {
        for(int j=0;j<2;++j)
        {
            for(int k=0;k<25;++k)
            {
                printf("%d ",packed[i*50+j*25+k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
    for(int i=0;i<6;++i)
    {
        for(int j=0;j<2;++j)
        {
            for(int k=0;k<25;++k)
            {
                printf("%d ",packed_neon[i*50+j*25+k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
    for(int i=0;i<6;++i)
    {
        for(int j=0;j<40;++j)
            printf("%d ",unpacked[i]);
        printf("\n");
    }
    printf("\n");

    for(int i=0;i<6;++i)
    {
        for(int j=0;j<40;++j)
            printf("%d ",unpacked_neon[i]);
        printf("\n");
    }
    printf("\n");
    return ;
}

