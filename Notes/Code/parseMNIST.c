#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define RESOLUTION_Y 28
#define RESOLUTION_X 28
// #define TRAING_DATA_PATH "../../TrainingData/train-images-idx3-ubyte/train-images.idx3-ubyte"
#define TRAING_DATA_PATH "image.idx"


void main()
{
    /*     LOAD IMAGE DATA     */
    FILE * trainingData = fopen("./image.idx", "rb");

    if (trainingData == NULL)
    {
        printf("\nERROR OPENING TRAINING DATA!!!\n");
    }
    else
    {
        printf("Opened training data.\n");
    }

    uint32_t magicNumber, imageCount, imageRows, imageCols;
    fread(&magicNumber, 4, 1, trainingData);
    fread(&imageCount, 4, 1, trainingData);
    fread(&imageCols, 4, 1, trainingData);
    fread(&imageRows, 4, 1, trainingData);
    magicNumber = __builtin_bswap32(magicNumber);
    imageCount = __builtin_bswap32(imageCount);
    imageCols = __builtin_bswap32(imageCols);
    imageRows = __builtin_bswap32(imageRows);

    printf("The training data contains:\n  %d images\n  Resolution %d x %d\n", imageCount, imageCols, imageRows);
    printf("Allocating %d bytes of memory.\n", imageCols*imageCols*imageCount);
    uint8_t (*image)[imageRows][imageCols] = malloc(imageCount * sizeof(*image));

    printf("Loading data into memory...\n");
    for(uint32_t imageIndex=0; imageIndex < imageCount; imageIndex++ )
    {
        for(uint32_t rowIndex=0; rowIndex < imageRows; rowIndex++)
        {
            for(uint32_t columIndex=0; columIndex < imageCols; columIndex++)
            {
                fread(&image[imageIndex][rowIndex][columIndex], 1, 1, trainingData);
            }
        }
    }
    printf("DONE\n");



    /*     LOAD LABEL DATA     */
    FILE * labelData = fopen("./labels.idx","rb");

    if (labelData == NULL)
    {
        printf("\nERROR OPENING LABEL DATA!!!\n");
    }
    else
    {
        printf("Opened label data.\n");
    }

    uint32_t label_magicNumber, labelCount;
    fread(&label_magicNumber, 4, 1, labelData);
    fread(&labelCount, 4, 1, labelData);
    label_magicNumber = __builtin_bswap32(label_magicNumber);
    labelCount = __builtin_bswap32(labelCount);

    printf("The label data contains:\n  %d labels\n", labelCount);
    printf("Alloc label mem\n");
    char labels[labelCount];

    for(uint32_t labelIndex = 0; labelIndex < labelCount; labelIndex++)
    {
        fread(&labels[labelIndex], 1, 1, labelData);
    }
    

    /*    PRINTING IMAGES TO SCREEN    */
    uint32_t chosenImage = 1;
    for(int blah=0; blah < 10; blah++)
    {
        chosenImage=blah;
        for(uint8_t printRow = 0; printRow < imageRows; printRow++)
        {
            printf("\n");
            for(uint8_t printCol = 0; printCol < imageCols; printCol++)
            {
                if (image[chosenImage][printRow][printCol] > 128)
                {
                    printf("▇");
                }
                else
                {
                    printf(" ");
                }


            }
            printf("\n");
        }
        printf("The solution is %d\n", labels[chosenImage]);
    }
    
    printf("Giving the memory back.\n");
    free(image);
    printf("END\n");
}
