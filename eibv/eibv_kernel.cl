__kernel void inverse(__global float* input_data1, __global float* input_data2, __global float* output_data)
{
    unsigned int i = get_global_id(0);

    output_data[i] = input_data1[i] * input_data2[i];
}
