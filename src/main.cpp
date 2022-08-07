#include<stdio.h>
int main()
{   int *data = new int[4];
    int *data1 = new int[4];

    for (int i = 0; i < 4; i++) {
        data[i] = i - 2;
        data1[i] = i - 2;
        printf("1 data[%d]=%d data1[%d]=%d", i, data[i], i, data1[i]);
    }

    asm volatile(
            "VLDM %0,{q0}\t\n"
            "VLDM %1,{q1}\t\n"
            "VADD.S32 q0,q0,q1\t\n"
            "VSUB.I32 q1,q0,q1\t\n"
            "VSTM %0,{q0}\t\n"
            "VSTM %1,{q1}\t\n"
    :"+r"(data),//%0
    "+r"(data1) //%1
    :
    : "memory", "q0", "q1"
    );

    for (int i = 0; i < 4; i++) {
        printf("2 data[%d]=%d data1[%d]=%d", i, data[i], i, data1[i]);
    }
    return 0;
}