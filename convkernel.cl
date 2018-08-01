////////////////////////////Image Filter//////////////////////////////
__kernel void convolution(__read_only image2d_t src, 
                        __write_only image2d_t dst,
                        int cols,
                        int rows,
                        __global int* filter,
                        int filterSize,
                        const sampler_t sampler)
{
    int w = get_global_id(0);
    int h = get_global_id(1);

    int2 coords;

    int halfWidth = (int)(filterSize / 2);

    uint4 sum = {0, 255, 0, 0};
    int color = 0;
    
    //Iterator for filter
    int filterIdx = 0;

    //sum filterdata
    for (int i = -halfWidth; i <= halfWidth; i++) {
        //{x, y}
        coords.y = h + i;
        for (int j = -halfWidth; j <= halfWidth; j++) {
            coords.x = w + j;
            uint4 pixel;

            pixel = read_imageui(src, sampler, coords);

            color += (int)pixel.x * filter[filterIdx];
            filterIdx++;
        }
    }

    sum.x = abs(color);
    //copy the result to output image
    if (h < rows && w < cols) {
        coords.x = w;
        coords.y = h;
        //sum = read_imageui(src, sampler, coords);
        write_imageui(dst, coords, sum);
    }

}
