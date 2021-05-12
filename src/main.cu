#include <cmath>		/* pow() */
#include <cstdint>		/* uint64_t */
#include <ctime>		/* time() */

#include <unistd.h>
#include <iostream>

#include "../include/ntt.h"	/* naiveNTT(), outOfPlaceNTT_DIT() */
#include "../include/utils.cuh"	/* printVec() */

using namespace std;

int main(int argc, char *argv[]){
 
  uint64_t batchSize = 1024;	
  uint64_t n = 4096;
  uint64_t p = 68719403009;
  uint64_t r = 36048964756;
  uint64_t* twiddleFactorArray = NULL ;
  twiddleFactorArray = preComputeTwiddleFactor(n,p,r) ;
//  uint64_t vec[n];
  uint64_t* vec = (uint64_t*)malloc(n*batchSize*sizeof(uint64_t));
 
  for(uint64_t i=0;i < batchSize;i++){
  	for (uint64_t j = 0; j < n; j++){
    		vec[i*n+j] = j ;
	}
  }

  uint64_t *outVec = inPlaceNTT_DIT(vec,batchSize,n,p,r,twiddleFactorArray);

//	printVec(outVec, n);

	return 0;

}
