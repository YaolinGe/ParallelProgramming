// CPP program to demonstrate the tanh()
// function when a string is passed as argument
#include <stdio.h>      /* printf */
#include <math.h>       /* tanh, log */

int main ()
{
  double param, result;
  param = log(2.0);
  result = tanh (param);
  printf ("The hyperbolic tangent of %f is %f.\n", param, result);
  return 0;
}
