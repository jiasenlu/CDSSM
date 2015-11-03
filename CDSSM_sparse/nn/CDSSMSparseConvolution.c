#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/CDSSMSparseConvolution.c"
#else

#ifdef _OPENMP
#include <omp.h>
#endif

static int nn_(checkInput4)(THTensor* t) {
  return t->nDimension == 2 && t->size[1] == 4;
}

static int nn_(checkSize3D)(THTensor* t, long size0, long size1, long size2) {
  return t->nDimension == 3 && t->size[0] == size0 && t->size[1] == size1 && t->size[2] == size2;
}

static int nn_(CDSSMSparseConvolution_updateOutput)(lua_State *L)
{
  long i, j;
  THTensor * input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor * weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor * bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor * output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  int batchSize = luaT_getfieldcheckint(L, 1, "batchSize");
  int winSize = luaT_getfieldcheckint(L, 1, "winSize");
  
  //printf("%d\n", batchSize);

  long outDim = weight->size[0];
  long inDim = weight->size[1];

  luaL_argcheck(L, nn_(checkInput4)(input), 2, "input size must be nnz x 4");
  luaL_argcheck(L, nn_(checkSize3D)(output, (long)batchSize, (long)winSize, outDim), 1, "output size wrong");
  luaL_argcheck(L, nn_(checkSize1D)(bias, outDim), 1, "bias size wrong");

  lua_getfield(L, 1, "shardBuffer");
  if (!lua_isnil(L, -1)) {
    THTensor *buffer =
      luaT_getfieldcheckudata(L, 1, "shardBuffer", torch_Tensor);
    long num_shards = buffer->size[3];
    luaL_argcheck(L,
                  buffer->nDimension == 4 && buffer->size[2] == outDim &&
                      num_shards > 0 && buffer->size[0] == (long)batchSize &&
                      buffer->size[1] == (long)winSize,
                  1,
                  "shardBuffer size wrong");

    THTensor_(zero)(buffer);
    #pragma omp parallel for private(i) schedule(static) num_threads(num_shards)
    for (i = 0; i < input->size[0]; i++) {
#ifdef _OPENMP
      int shardId = omp_get_thread_num();
#else
      int shardId = 0;
#endif
      long idx = (long)(THTensor_(get2d)(input, i, 0)) - 1;
      long win = (long)(THTensor_(get2d)(input, i, 1)) - 1;
      long offset = (long)(THTensor_(get2d)(input, i, 2)) - 1;
      if (offset >= 0 && offset < inDim) {
        THBlas_(axpy)(outDim,
                      THTensor_(get2d)(input, i, 3),
                      THTensor_(data)(weight) + offset * weight->stride[1],
                      weight->stride[0],
                      THTensor_(data)(buffer) + idx * buffer->stride[0] + win * buffer->stride[1] + shardId * buffer->stride[3],
                      buffer->stride[2]);
      } else {
        luaL_error(L, "index out of bound. updateOutput: \
%ld not between 1 and %ld", offset + 1, inDim);
      }
    }

    THTensor_(sum)(output, buffer, 3);
    THTensor_(resize3d)(output, batchSize, winSize, outDim);
  for(i=0; i<batchSize; i++)
    for(j=0; j<winSize; j++)
    {

    THBlas_(axpy)(outDim, 
                  1,
                  THTensor_(data)(bias),
                  bias->stride[0],
                  THTensor_(data)(output) + i * output->stride[0] + j*output->stride[1],
                  output->stride[2]);
    }
    lua_getfield(L, 1, "output");
    return 1;
  }
  for(i = 0; i < input->size[0]; i++)
  {
    long idx = (long)(THTensor_(get2d)(input, i, 0)) - 1;
    long win = (long)(THTensor_(get2d)(input, i, 1)) - 1;
    long offset = (long)(THTensor_(get2d)(input, i, 2)) - 1;

    if(offset >= 0 && offset < inDim) /* make sure indices are in bounds.. */
    {
        real val = THTensor_(get2d)(input, i, 3);
        THBlas_(axpy)(output->size[1],
                      val,
                      THTensor_(data)(weight)+offset*weight->stride[1],
                      weight->stride[0],
                      THTensor_(data)(output)+idx*output->stride[0] + win*output->stride[1],
                      output->stride[2]);
    }
    else {
        luaL_error(L, "index out of bound. updateOutput: \
%ld not between 1 and %ld", offset + 1, inDim);
    }
  }
  for(i=0; i<batchSize; i++)
    for(j=0; j<winSize; j++)
    {

    THBlas_(axpy)(outDim, 
                  1,
                  THTensor_(data)(bias),
                  bias->stride[0],
                  THTensor_(data)(output) + i * output->stride[0] + j*output->stride[1],
                  output->stride[2]);
    }
  lua_getfield(L, 1, "output");
  return 1;
}

static int nn_(CDSSMSparseConvolution_accGradParameters)(lua_State *L)
{
  long i;
  THTensor * input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor * gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real scale = luaL_optnumber(L, 4, 1);
  THTensor * bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor * weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor * gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  THTensor * gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  real weightDecay = luaT_getfieldchecknumber(L, 1, "weightDecay");
  int batchSize = luaT_getfieldcheckint(L, 1, "batchSize");
  int winSize = luaT_getfieldcheckint(L, 1, "winSize");

  long nnz = input->size[0];
  long outDim = weight->size[0];
  long inDim = weight->size[1];

  luaL_argcheck(L, nn_(checkInput4)(input), 2, "input size must be nnz x 4");
  luaL_argcheck(
    L, nn_(checkSize3D)(gradOutput, (long)batchSize, (long)winSize, outDim), 3, "gradOutput size wrong");
  luaL_argcheck(
    L, nn_(checkSize2D)(gradWeight, outDim, inDim), 1, "gradWeight size wrong");
  luaL_argcheck(
    L, nn_(checkSize1D)(gradBias, outDim), 1, "gradBias size wrong");

  #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 100000)
  for(i = 0; i < nnz; i++)
  {
      long idx = (long)(THTensor_(get2d)(input, i, 0)) - 1;
      long win = (long)(THTensor_(get2d)(input, i, 1)) - 1;
      long offset = (long)(THTensor_(get2d)(input, i, 2)) - 1;

      if(offset >= 0 && offset < inDim) /* make sure indices are in bounds.. */
      {
          real val = scale*THTensor_(get2d)(input, i, 3);

          THBlas_(axpy)(outDim,
                        val,
                        THTensor_(data)(gradOutput)+idx*gradOutput->stride[0]+win*gradOutput->stride[1],
                        gradOutput->stride[2],
                        THTensor_(data)(gradWeight)+offset*gradWeight->stride[1],
                        gradWeight->stride[0]);
      }
      else {
          luaL_error(L, "index out of bound. accGradParameters: \
%ld not between 1 and %ld", offset + 1, inDim);
      }
  }

  THTensor_(sum)(gradOutput, gradOutput, 0);
  THTensor_(sum)(gradOutput, gradOutput, 1);
  THTensor_(resize1d)(gradOutput, outDim);

  THTensor_(cadd)(gradBias, gradBias, scale, gradOutput);

  if(weightDecay != 0) {
    #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 100000)
    for(i = 0; i < nnz; i++) {
      long offset = (long)(THTensor_(get2d)(input, i, 2)) - 1;
      THBlas_(axpy)(outDim,
                    weightDecay,
                    THTensor_(data)(weight) + offset*weight->stride[1],
                    weight->stride[0],
                    THTensor_(data)(gradWeight)+offset*gradWeight->stride[1],
                    gradWeight->stride[0]);
    }

    THTensor_(cadd)(gradBias, gradBias, weightDecay, bias);
  }

  return 0;
}

int nn_(CDSSMSparseConvolution_updateParameters)(lua_State *L)
{
  long i;
  real learningRate = luaL_checknumber(L, 2);
  THTensor * weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor * bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor * gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  THTensor * gradWeight = luaT_getfieldcheckudata(
    L, 1, "gradWeight", torch_Tensor);
  THTensor * lastInput = luaT_getfieldcheckudata(
    L, 1, "lastInput", torch_Tensor);

  long nnz = lastInput->size[0];
  long outDim = weight->size[0];
  long inDim = weight->size[1];

  luaL_argcheck(
    L, nn_(checkSize2D)(gradWeight, outDim, inDim), 1, "gradWeight size wrong");
  luaL_argcheck(
    L, nn_(checkSize1D)(bias, outDim), 1, "bias size wrong");
  luaL_argcheck(
    L, nn_(checkSize1D)(gradBias, outDim), 1, "gradBias size wrong");

  THTensor_(cadd)(bias, bias, -learningRate, gradBias);

  #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 50000)
  for(i = 0; i < nnz; i++)
  {
      long offset = (long)(THTensor_(get2d)(lastInput, i, 2)) - 1;

      if(offset >= 0 && offset < inDim) /* make sure indices are in bounds.. */
      {
          real* pGradWeight =
            THTensor_(data)(gradWeight)+offset*gradWeight->stride[1];
          THBlas_(axpy)(outDim,
                        -learningRate,
                        pGradWeight,
                        gradWeight->stride[0],
                        THTensor_(data)(weight)+offset*weight->stride[1],
                        weight->stride[0]);
      }
      else {
          luaL_error(L, "index out of bound. updateParameters: \
%ld not between 1 and %ld", offset + 1, inDim);
      }
  }
  return 0;
}

int nn_(CDSSMSparseConvolution_zeroGradParameters)(lua_State *L)
{
  long i;
  THTensor * gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  THTensor * gradWeight = luaT_getfieldcheckudata(
    L, 1, "gradWeight", torch_Tensor);
  THTensor * lastInput = luaT_getfieldcheckudata(
    L, 1, "lastInput", torch_Tensor);

  long nnz = lastInput->size[0];
  long outDim = gradWeight->size[0];
  long inDim = gradWeight->size[1];

  luaL_argcheck(
    L, nn_(checkSize1D)(gradBias, outDim), 1, "gradBias size wrong");

  THTensor_(zero)(gradBias);
  #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 50000)
  for(i = 0; i < nnz; i++)
  {
      long offset = (long)(THTensor_(get2d)(lastInput, i, 2)) - 1;

      if(offset >= 0 && offset < inDim) /* make sure indices are in bounds.. */
      {
          real* pGradWeight =
            THTensor_(data)(gradWeight)+offset*gradWeight->stride[1];
          if(gradWeight->stride[0] == 1) {
              THVector_(fill)(pGradWeight, 0, outDim);
          } else {
              long j;
              for(j = 0; j < outDim; ++j) {
                  pGradWeight[j * gradWeight->stride[0]] = 0;
              }
          }
      }
      else {
          luaL_error(L, "index out of bound. zeroGradParameters: \
%ld not between 1 and %ld", offset + 1, inDim);
      }
  }
  return 0;
}

static int nn_(CDSSMSparseConvolution_updateGradInput)(lua_State *L) {
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput =
      luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int batchSize = luaT_getfieldcheckint(L, 1, "batchSize");
  int winSize = luaT_getfieldcheckint(L, 1, "winSize");

  long i;
  long nnz = input->size[0];
  long outDim = weight->size[0];
  long inDim = weight->size[1];

  luaL_argcheck(
    L, nn_(checkInput4)(input), 2, "input must be an nnz x 4 tensor");
  luaL_argcheck(
    L, nn_(checkSize3D)(gradOutput, (long)batchSize, (long)winSize, outDim), 3, "gradOutput size wrong");

  THTensor_(resize2d)(gradInput, input->size[0], input->size[1]);

  #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 100000)
  for (i = 0; i < nnz; ++i) {
    long idx = (long)(THTensor_(get2d)(input, i, 0)) - 1;
    long win = (long)(THTensor_(get2d)(input, i, 1)) - 1;
    long offset = (long)(THTensor_(get2d)(input, i, 2)) - 1;
    THTensor_(set2d)(gradInput, i, 0, idx + 1);
    THTensor_(set2d)(gradInput, i, 1, win + 1);    
    THTensor_(set2d)(gradInput, i, 2, offset + 1);

    if (offset >= 0 && offset < inDim) {
      real val =
          THBlas_(dot)(outDim,
                       THTensor_(data)(gradOutput) + idx * gradOutput->stride[0] + win*gradOutput->stride[1],
                       gradOutput->stride[2],
                       THTensor_(data)(weight) + offset * weight->stride[1],
                       weight->stride[0]);
      THTensor_(set2d)(gradInput, i, 3, val);
    } else {
      luaL_error(L, "index out of bound. updateGradInput: \
%ld not between 1 and %ld", offset + 1, inDim);
    }
  }
  return 0;
}

static const struct luaL_Reg nn_(CDSSMSparseConvolution__) [] = {
  {"CDSSMSparseConvolution_updateOutput", nn_(CDSSMSparseConvolution_updateOutput)},
  {"CDSSMSparseConvolution_accGradParameters", nn_(CDSSMSparseConvolution_accGradParameters)},
  {"CDSSMSparseConvolution_updateParameters", nn_(CDSSMSparseConvolution_updateParameters)},
  {"CDSSMSparseConvolution_zeroGradParameters", nn_(CDSSMSparseConvolution_zeroGradParameters)},
  {"CDSSMSparseConvolution_updateGradInput", nn_(CDSSMSparseConvolution_updateGradInput)},
  {NULL, NULL}
};

void nn_(CDSSMSparseConvolution_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(CDSSMSparseConvolution__), "nn");
  lua_pop(L,1);
}

#endif
