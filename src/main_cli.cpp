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
  // int * out_num_iters = NULL;
  // const char* parameter = "transform_file=tmp_transform header_file=/mnt/chjinche/projects/LightGBM/tests/data/clean_header";
  // BoosterHandle* handle = NULL;
  // // LGBM_BoosterCreateFromModelfile(
  // //   "/mnt/chjinche/projects/LightGBM/tests/data/trained_model",
  // //   out_num_iters, handle,
  // //   "tmp_transform");
  // LGBM_BoosterPredictForFile(
  //                   handle,
  //                   "/mnt/chjinche/projects/LightGBM/tests/data/clean_input_parts_0",
  //                   0,
  //                   1,
  //                   0,
  //                   5,
  //                   parameter,
  //                   "");
  

  int argc = 2;
  // char* argv[] = {"../../lightgbm", "config=/mnt/chjinche/projects/LightGBM/examples/multiclass_classification/train.conf"};
  // char* argv[] = {"../../lightgbm", "config=/mnt/chjinche/projects/LightGBM/tests/data/l2_train.conf"};
  // char* argv[] = {"../../lightgbm", "config=/mnt/chjinche/projects/LightGBM/tests/cpp_tests/predict.conf"};
  char* argv[] = {"../../lightgbm", "config=/mnt/chjinche/projects/LightGBM/tests/data/l2_predict.conf"};
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
