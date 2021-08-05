/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/boosting.h>
#include <boost/algorithm/string/trim.hpp>

#include "dart.hpp"
#include "gbdt.h"
#include "goss.hpp"
#include "rf.hpp"

namespace LightGBM {

std::string GetBoostingTypeFromModelFile(const char* filename) {
  TextReader<size_t> model_reader(filename, true);
  std::string type = model_reader.first_line();
  return type;
}

bool Boosting::LoadFileToBoosting(Boosting* boosting, const char* filename) {
  auto start_time = std::chrono::steady_clock::now();
  if (boosting != nullptr) {
    TextReader<size_t> model_reader(filename, true);
    size_t buffer_len = 0;
    auto buffer = model_reader.ReadContent(&buffer_len);
    if (!boosting->LoadModelFromString(buffer.data(), buffer_len)) {
      return false;
    }
  }
  std::chrono::duration<double, std::milli> delta = (std::chrono::steady_clock::now() - start_time);
  Log::Debug("Time for loading model: %f seconds", 1e-3*delta);
  return true;
}

Boosting* Boosting::CreateBoosting(const std::string& type, const char* filename, const char* transform_filename) {
  // save model copy.
  std::string core_model_filename(filename);
  if (filename == nullptr || filename[0] == '\0') {
    if (type == std::string("gbdt")) {
      return new GBDT();
    } else if (type == std::string("dart")) {
      return new DART();
    } else if (type == std::string("goss")) {
      return new GOSS();
    } else if (type == std::string("rf")) {
      return new RF();
    } else {
      return nullptr;
    }
  } else {
    std::unique_ptr<Boosting> ret;
    // split model file to transform and real model file.
    if (GetBoostingTypeFromModelFile(filename) == std::string("transform")){
      Log::Info("The model has transform section. Splitting ...");
      //TODO: verify transform_file is given.
      std::ifstream fin(filename);
      std::string line;
      std::vector<std::string> transform_lines, model_lines;
      bool is_transform = true;
      while (std::getline(fin, line)){
        boost::trim(line);
        if (line == std::string("tree"))
          is_transform = false;
        if (is_transform)
          transform_lines.push_back(line);
        else
          model_lines.push_back(line);
      }
      fin.close();
      std::ofstream tfout(transform_filename);
      for(auto str : transform_lines)
        tfout << str << std::endl;
      tfout.close();

      // TODO: to be improved, cause tmpnam is dangerous.
      core_model_filename = std::tmpnam(nullptr);
      Log::Info("Created a tmp model file: %s", core_model_filename.c_str());
      std::ofstream mfout(core_model_filename);
      for(auto str : model_lines)
        mfout << str << std::endl;
      mfout.close();
    }
    if (GetBoostingTypeFromModelFile(core_model_filename.c_str()) == std::string("tree")) {
      if (type == std::string("gbdt")) {
        ret.reset(new GBDT());
      } else if (type == std::string("dart")) {
        ret.reset(new DART());
      } else if (type == std::string("goss")) {
        ret.reset(new GOSS());
      } else if (type == std::string("rf")) {
        return new RF();
      } else {
        Log::Fatal("Unknown boosting type %s", type.c_str());
      }
      LoadFileToBoosting(ret.get(), core_model_filename.c_str());
    } else {
      Log::Fatal("Unknown model format or submodel type in model file %s", filename);
    }
    return ret.release();
  }
}

}  // namespace LightGBM
