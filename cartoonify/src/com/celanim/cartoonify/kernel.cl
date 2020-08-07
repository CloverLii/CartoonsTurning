#define COLOUR_BITS 8
#define RED 2
#define GREEN 1
#define BLUE 0

__constant int GAUSSIAN_FILTER[] = {
		2,  4,  5,  4,  2, // sum=17
		4,  9, 12,  9,  4, // sum=38
		5, 12, 15, 12,  5, // sum=49
		4,  9, 12,  9,  4, // sum=38
		2,  4,  5,  4,  2  // sum=17
	};
	
__constant float GAUSSIAN_SUM = 159.0;
__constant int numColours = 3;
__constant int COLOUR_MASK = (1 << COLOUR_BITS) - 1;

__constant int SOBEL_VERTICAL_FILTER[] = {
		-1,  0, +1,
		-2,  0, +2,
		-1,  0, +1
};

__constant int SOBEL_HORIZONTAL_FILTER[] = {
		+1, +2, +1,
		0,   0,  0,
		-1, -2, -1
};

// Restricts an index to be within the image
static int wrap(int pos, int size) 
{
	if (pos < 0) {
		pos = -1 - pos;
	} else if (pos >= size) {
		pos = (size - 1) - (pos - size);
	}

	return pos;
}

// Clamp a colour value to be within the allowable range for each colour	
static int clamp2(float value) 
{
	int result = (int) (value + 0.5); // round to nearest integer
	if (result <= 0) {
		return 0;
	} else if (result > COLOUR_MASK) {
		return 255;
	} else {
		return result;
	}
}	

//Extract a given colour channel out of the given pixel	
static int colourValue(int pixel, int colour) 
{
	return (pixel >> (colour * COLOUR_BITS)) & COLOUR_MASK;
}

//Constructs one integer RGB pixel from the individual components
static int createPixel(int redValue, int greenValue, int blueValue)
{

	return (redValue << (2 * COLOUR_BITS)) + (greenValue << COLOUR_BITS) + blueValue;
}

//Get the red value of the given pixel
static int red(int pixel)
{
	return colourValue(pixel, RED);
}

//Get the green value of the given pixel
static int green(int pixel)
{
	return colourValue(pixel, GREEN);
}

//Get the blue value of the given pixel
static int blue(int pixel) 
{
	return colourValue(pixel, BLUE);
}

// This applies the given N*N filter around the pixel (xCentre,yCentre)
static int convolution(int xCentre, int yCentre, int width, int height,
				__global int *curPixels){						
		int rSum = 0;
		int gSum = 0;
		int bSum = 0;
		
		int filterSize = 5;
		int filterHalf = filterSize / 2;				
		for (int filterY = 0; filterY < filterSize; filterY++) {
			int y = wrap(yCentre + filterY - filterHalf, height);
			
			for (int filterX = 0; filterX < filterSize; filterX++) {
				int x = wrap(xCentre + filterX - filterHalf, width);
				
				int rgb = curPixels[y * width + x];							
				int filterVal = GAUSSIAN_FILTER[filterY * filterSize + filterX];	
				rSum += red(rgb) * filterVal;
				gSum += green(rgb) * filterVal;
				bSum += blue(rgb) * filterVal;
			}
		}
				int red = clamp2(rSum / GAUSSIAN_SUM);
				int green = clamp2(gSum / GAUSSIAN_SUM);
				int blue = clamp2(bSum / GAUSSIAN_SUM);				
				return createPixel(red, green, blue);
	}
	
// Adds one new image that is a blurred version of the current image	
__kernel void gaussianBlur(const int width, const int height, 
						__global int *curPixels,
						__global int *output){
		
		int gid = get_global_id(0);
		int y = gid / width;
		int x = gid % width;	
		output[gid] = convolution(x, y, width, height, curPixels);
	}	

// This applies the given N*N filter around the pixel (xCentre,yCentre)	
static int convolution2(int xCentre, int yCentre, int width, int height, int edgeThreshold,
				__global int *curPixels){
		
		int vRed = 0;
		int vGreen = 0;
		int vBlue = 0;
		int hRed = 0;
		int hGreen = 0;
		int hBlue = 0;
						
		int filterSize = 3;
		int filterHalf = filterSize / 2;				
		for (int filterY = 0; filterY < filterSize; filterY++) {
			int y = wrap(yCentre + filterY - filterHalf, height);
			
			for (int filterX = 0; filterX < filterSize; filterX++) {
				int x = wrap(xCentre + filterX - filterHalf, width);
				
				int rgb = curPixels[y * width + x];							
				int filterValV = SOBEL_VERTICAL_FILTER[filterY * filterSize + filterX];	
				vRed += red(rgb) * filterValV;
				vGreen += green(rgb) * filterValV;
				vBlue += blue(rgb) * filterValV;
				
				int filterValH = SOBEL_HORIZONTAL_FILTER[filterY * filterSize + filterX];
				hRed += red(rgb) * filterValH;
				hGreen += green(rgb) * filterValH;
				hBlue += blue(rgb) * filterValH;	
			}
		}
		int verticalGradient = abs(vRed) + abs(vGreen) + abs(vBlue);
		int horizontalGradient = abs(hRed) + abs(hGreen) + abs(hBlue);
		int totalGradient = verticalGradient + horizontalGradient;
		if (totalGradient >= edgeThreshold) {
			return createPixel(0, 0, 0); 
		} else {
			return createPixel(COLOUR_MASK, COLOUR_MASK, COLOUR_MASK);
		}
}

// Detects edges in the current image and adds an image where black pixels	
__kernel void sobelEdgeDetect(int width, int height, int edgeThreshold,
								__global int *curPixels,
								__global int *output){
						
		int gid = get_global_id(0);
		int y = gid / width;
		int x = gid % width;
		output[gid] = convolution2(x, y, width, height, edgeThreshold, curPixels);

	}	

// Converts the given colour value (eg. 0..255) to an approximate colour value
static int quantizeColour(int colourValue, int numPerChannel){
	float colour = colourValue / (COLOUR_MASK + 1.0f) * numPerChannel;
	int discrete = round(colour - 0.49999f);
	int newColour = discrete * COLOUR_MASK / (numPerChannel - 1);
	return newColour;
}

//Adds a new image that is the same as the current image but with fewer colours	
__kernel void reduceColours(__global int *curPixels,
							__global int *output) {
						
		int gid = get_global_id(0);
		int rgb = curPixels[gid];
		
		int newRed = quantizeColour(red(rgb), numColours);
		int newGreen = quantizeColour(green(rgb), numColours);
		int newBlue = quantizeColour(blue(rgb), numColours);
		int newRGB = createPixel(newRed, newGreen, newBlue);
		output[gid] = newRGB;	
	}	
	
// Merges a mask image on top of another image	
__kernel void mergeMask(__global int *edgePixels,
						__global int *colorPixels,
						__global int *output){
						
		int gid = get_global_id(0);	
		int white = createPixel(COLOUR_MASK, COLOUR_MASK, COLOUR_MASK);	
		if (colorPixels[gid] == white) {
			output[gid] = edgePixels[gid];
		} else {
			output[gid] = colorPixels[gid];
		}
	}