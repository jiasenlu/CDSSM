#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SparseLinearBatch.c"
#else

#ifdef _OPENMP
#include <omp.h>
#endif

static int nn_(checkInput3D)(THTensor* t) {
  return t->nDimension == 2 && t->size[1] == 3;
}

static int nn_(SparseLinearBatch_updateOutput)(lua_State *L)
{
  long i;
  THTensor * input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor * weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor * bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor * output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  int batchSize = luaT_getfieldcheckint(L, 1, "batchSize");
  //printf("%d\n", batchSize);

  long outDim = weight->size[0];
  long inDim = weight->size[1];

  luaL_argcheck(L, nn_(checkInput3D)(input), 2, "input size must be nnz x 3");
  luaL_argcheck(L, nn_(checkSize2D)(output, (long)batchSize, outDim), 1, "output size wrong");
  luaL_argcheck(L, nn_(checkSize1D)(bias, outDim), 1, "bias size wrong");

  lua_getfield(L, 1, "shardBuffer");
  if (!lua_isnil(L, -1)) {
    THTensor *buffer =
      luaT_getfieldcheckudata(L, 1, "shardBuffer", torch_Tensor);
    long num_shards = buffer->size[2];
    luaL_argcheck(L,
                  buffer->nDimension == 3 && buffer->size[1] == outDim &&
                      num_shards > 0 && buffer->size[0] == (long)batchSize,
                  1,
                  "shardBuffer size wrong");

    THTensor_(zero)(buffer);
    #pragma omp parallel for private(i) schedule(static) num_threads(num_shards)
    for (i = 0; i < input->size[0]; i++) {
#ifdef _OPENMP
      int shardId = omp_get_thread_num();
#else
      int shardId = 1;
#endif
      long idx = (long)(THTensor_(get2d)(input, i, 0)) - 1;
      long offset = (long)(THTensor_(get2d)(input, i, 1)) - 1;

      if (offset >= 0 && offset < inDim) {
        THBlas_(axpy)(outDim,
                      THTensor_(get2d)(input, i, 2),
                      THTensor_(data)(weight) + offset * weight->stride[1],
                      weight->stride[0],
                      THTensor_(data)(buffer) + idx * buffer->stride[0] + shardId * buffer->stride[2],
                      buffer->stride[1]);
      } else {
        luaL_error(L, "index out of bound. updateOutput: \
%ld not between 1 and %ld", offset + 1, inDim);
      }
    }

    THTensor_(sum)(output, buffer, 2);
    THTensor_(resize2d)(output, batchSize, outDim);
    for(i=0; i<batchSize; i++)
    {

      THBlas_(axpy)(outDim, 
                    1,
                    THTensor_(data)(bias),
                    bias->stride[0],
                    THTensor_(data)(output) + i * output->stride[0],
                    output->stride[1]);
    }

    lua_getfield(L, 1, "output");
    return 1;
  }

  for(i = 0; i < input->size[0]; i++)
  {
    long idx = (long)(THTensor_(get2d)(input, i, 0)) - 1;
    long offset = (long)(THTensor_(get2d)(input, i, 1)) - 1;

    if(offset >= 0 && offset < inDim) /* make sure indices are in bounds.. */
    {
        real val = THTensor_(get2d)(input, i, 2);
        THBlas_(axpy)(output->size[1],
                      val,
                      THTensor_(data)(weight)+offset*weight->stride[1],
                      weight->stride[0],
                      THTensor_(data)(output)+idx*output->stride[0],
                      output->stride[1]);
    }
    else {
        luaL_error(L, "index out of bound. updateOutput: \
%ld not between 1 and %ld", offset + 1, inDim);
    }
  }
  for(i=0; i<batchSize; i++)
  {

    THBlas_(axpy)(outDim, 
                  1,
                  THTensor_(data)(bias),
                  bias->stride[0],
                  THTensor_(data)(output) + i * output->stride[0],
                  output->stride[1]);
  }
  lua_getfield(L, 1, "output");
  return 1;
}

static int nn_(SparseLinearBatch_accGradParameters)(lua_State *L)
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

  long nnz = input->size[0];
  long outDim = weight->size[0];
  long inDim = weight->size[1];

  luaL_argcheck(L, nn_(checkInput3D)(input), 2, "input size must be nnz x 3");
  luaL_argcheck(
    L, nn_(checkSize2D)(gradOutput, (long)batchSize, outDim), 3, "gradOutput size wrong");
  luaL_argcheck(
    L, nn_(checkSize2D)(gradWeight, outDim, inDim), 1, "gradWeight size wrong");
  luaL_argcheck(
    L, nn_(checkSize1D)(gradBias, outDim), 1, "gradBias size wrong");

  #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 100000)
  for(i = 0; i < nnz; i++)
  {
      long idx = (long)(THTensor_(get2d)(input, i, 0)) - 1;
      long offset = (long)(THTensor_(get2d)(input, i, 1)) - 1;

      if(offset >= 0 && offset < inDim) /* make sure indices are in bounds.. */
      {
          real val = scale*THTensor_(get2d)(input, i, 2);

          THBlas_(axpy)(outDim,
                        val,
                        THTensor_(data)(gradOutput)+idx*gradOutput->stride[0],
                        gradOutput->stride[1],
                        THTensor_(data)(gradWeight)+offset*gradWeight->stride[1],
                        gradWeight->stride[0]);
      }
      else {
          luaL_error(L, "index out of bound. accGradParameters: \
%ld not between 1 and %ld", offset + 1, inDim);
      }
  }

  THTensor_(sum)(gradOutput, gradOutput, 0);
  THTensor_(resize1d)(gradOutput, outDim);

  THTensor_(cadd)(gradBias, gradBias, scale, gradOutput);

  if(weightDecay != 0) {
    #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 100000)
    for(i = 0; i < nnz; i++) {
      long offset = (long)(THTensor_(get2d)(input, i, 1)) - 1;
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

int nn_(SparseLinearBatch_updateParameters)(lua_State *L)
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
      long offset = (long)(THTensor_(get2d)(lastInput, i, 1)) - 1;

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

int nn_(SparseLinearBatch_zeroGradParameters)(lua_State *L)
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
      long offset = (long)(THTensor_(get2d)(lastInput, i, 1)) - 1;

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

static int nn_(SparseLinearBatch_updateGradInput)(lua_State *L) {
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput =
      luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int batchSize = luaT_getfieldcheckint(L, 1, "batchSize");

  long i;
  long nnz = input->size[0];
  long outDim = weight->size[0];
  long inDim = weight->size[1];

  luaL_argcheck(
    L, nn_(checkInput3D)(input), 2, "input must be an nnz x 3 tensor");
  luaL_argcheck(
    L, nn_(checkSize2D)(gradOutput, (long)batchSize, outDim), 3, "gradOutput size wrong");

  THTensor_(resize2d)(gradInput, input->size[0], input->size[1]);

  #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 100000)
  for (i = 0; i < nnz; ++i) {
    long idx = (long)(THTensor_(get2d)(input, i, 0)) - 1;
    long offset = (long)(THTensor_(get2d)(input, i, 1)) - 1;
    THTensor_(set2d)(gradInput, i, 0, idx + 1);
    THTensor_(set2d)(gradInput, i, 1, offset + 1);

    if (offset >= 0 && offset < inDim) {
      real val =
          THBlas_(dot)(outDim,
                       THTensor_(data)(gradOutput) + idx * gradOutput->stride[0],
                       gradOutput->stride[1],
                       THTensor_(data)(weight) + offset * weight->stride[1],
                       weight->stride[0]);
      THTensor_(set2d)(gradInput, i, 2, val);
    } else {
      luaL_error(L, "index out of bound. updateGradInput: \
%ld not between 1 and %ld", offset + 1, inDim);
    }
  }
  return 0;
}

static const struct luaL_Reg nn_(SparseLinearBatch__) [] = {
  {"SparseLinearBatch_updateOutput", nn_(SparseLinearBatch_updateOutput)},
  {"SparseLinearBatch_accGradParameters", nn_(SparseLinearBatch_accGradParameters)},
  {"SparseLinearBatch_updateParameters", nn_(SparseLinearBatch_updateParameters)},
  {"SparseLinearBatch_zeroGradParameters", nn_(SparseLinearBatch_zeroGradParameters)},
  {"SparseLinearBatch_updateGradInput", nn_(SparseLinearBatch_updateGradInput)},
  {NULL, NULL}
};

void nn_(SparseLinearBatch_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SparseLinearBatch__), "nn");
  lua_pop(L,1);
}

#endif
