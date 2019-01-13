#include "tpunit.h"

int main() {
   /**
    * Run all of the registered tpunit++ tests. Returns 0 if
    * all tests are successful, otherwise returns the number
    * of failing assertions.
    */
   return tpunit::Tests::Run();
}