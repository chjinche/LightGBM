/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/application.h>
#include <LightGBM/c_api.h>


#include <iostream>

#include "network/linkers.h"

// int main(int argc, char** argv) {
int main() {
  // int * out_num_iters;
  // *out_num_iters = 10;
  // BoosterHandle* out;
  // LGBM_BoosterCreateFromModelfile(
  //   "/mnt/chjinche/projects/LightGBM/tests/data/trained_model",
  //   out_num_iters, out,
  //   "tmp_transform");

  int argc = 2;
  // char* argv[] = {"../../lightgbm", "config=/mnt/chjinche/projects/LightGBM/examples/multiclass_classification/train.conf"};
  char* argv[] = {"../../lightgbm", "config=/mnt/chjinche/projects/LightGBM/tests/data/l2_train.conf"};
  // char* argv[] = {"../../lightgbm", "config=/mnt/chjinche/projects/LightGBM/tests/cpp_tests/predict.conf"};
  // char* argv[] = {"../../lightgbm", "config=/mnt/chjinche/projects/LightGBM/tests/data/l2_predict.conf"};
  bool success = false;
  try {
    LightGBM::Application app(argc, argv);
    app.Run();
 #ifdef USE_MPI
    LightGBM::Linkers::MpiFinalizeIfIsParallel();
#endif

    success = true;
  }
  catch (const std::exception& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex.what() << std::endl;
  }
  catch (const std::string& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex << std::endl;
  }
  catch (...) {
    std::cerr << "Unknown Exceptions" << std::endl;
  }

  if (!success) {
#ifdef USE_MPI
    LightGBM::Linkers::MpiAbortIfIsParallel();
#endif

    exit(-1);
  }
}
