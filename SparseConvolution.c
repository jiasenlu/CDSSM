#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SparseConvolution.c"
#else

#ifdef _OPENMP
#include <omp.h>
#endif

static int nn_(checkInput)(THTensor* t) {
  return t->nDimension == 2 && t->size[1] == 2;
}

static int nn_(checkSize2D)(THTensor* t, long size0, long size1) {
  return t->nDimension == 2 && t->size[0] == size0 && t->size[1] == size1;
}

static int nn_(checkSize1D)(THTensor* t, long size0) {
  return t->nDimension == 1 && t->size[0] == size0;
}

static int nn_(SparseConvolution_updateOutput)(lua_State *L)
{
    long i;
    THTensor * input = luaT_checkudata(L, 2, torch_Tensor);
    THTensor * weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
    THTensor * bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
    THTensor * output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

    long outDim = weight->size[0];
    long inDim = weight->size[1];
    
    luaL_argcheck(L, nn_(checkInput)(input), 2, "input size must be nnz x 2");
    luaL_argcheck(L, nn_(checkSize1D)(output, outDim), 1, "output size wrong");
    luaL_argcheck(L, nn_(checkSize1D)(bias, outDim), 1, "bias size wrong");





}