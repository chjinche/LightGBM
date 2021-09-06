/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_IO_PARSER_HPP_
#define LIGHTGBM_IO_PARSER_HPP_

#include <LightGBM/dataset.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>

#include <unordered_map>
#include <utility>
#include <vector>
#include "TransformProcessor.h"
#include "Logger.h"

namespace LightGBM {

class CSVParser: public Parser {
 public:
  explicit CSVParser(int label_idx, int total_columns, AtofFunc atof)
    :label_idx_(label_idx), total_columns_(total_columns), atof_(atof) {
  }
  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features, double* out_label) const override {
    int idx = 0;
    double val = 0.0f;
    int offset = 0;
    *out_label = 0.0f;
    while (*str != '\0') {
      str = atof_(str, &val);
      if (idx == label_idx_) {
        *out_label = val;
        offset = -1;
      } else if (std::fabs(val) > kZeroThreshold || std::isnan(val)) {
        out_features->emplace_back(idx + offset, val);
      }
      ++idx;
      if (*str == ',') {
        ++str;
      } else if (*str != '\0') {
        Log::Fatal("Input format error when parsing as CSV");
      }
    }
  }

  inline int NumFeatures() const override {
    return total_columns_ - (label_idx_ >= 0);
  }

 private:
  int label_idx_ = 0;
  int total_columns_ = -1;
  AtofFunc atof_;
};

class TSVParser: public Parser {
 public:
  explicit TSVParser(int label_idx, int total_columns, AtofFunc atof)
    :label_idx_(label_idx), total_columns_(total_columns), atof_(atof) {
  }
  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features, double* out_label) const override {
    int idx = 0;
    double val = 0.0f;
    int offset = 0;
    while (*str != '\0') {
      str = atof_(str, &val);
      if (idx == label_idx_) {
        *out_label = val;
        offset = -1;
      } else if (std::fabs(val) > kZeroThreshold || std::isnan(val)) {
        out_features->emplace_back(idx + offset, val);
      }
      ++idx;
      if (*str == '\t') {
        ++str;
      } else if (*str != '\0') {
        Log::Fatal("Input format error when parsing as TSV");
      }
    }
  }

  inline int NumFeatures() const override {
    return total_columns_ - (label_idx_ >= 0);
  }

 private:
  int label_idx_ = 0;
  int total_columns_ = -1;
  AtofFunc atof_;
};

class LibSVMParser: public Parser {
 public:
  explicit LibSVMParser(int label_idx, int total_columns, AtofFunc atof)
    :label_idx_(label_idx), total_columns_(total_columns), atof_(atof) {
    if (label_idx > 0) {
      Log::Fatal("Label should be the first column in a LibSVM file");
    }
  }
  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features, double* out_label) const override {
    int idx = 0;
    double val = 0.0f;
    if (label_idx_ == 0) {
      str = atof_(str, &val);
      *out_label = val;
      str = Common::SkipSpaceAndTab(str);
    }
    while (*str != '\0') {
      str = Common::Atoi(str, &idx);
      str = Common::SkipSpaceAndTab(str);
      if (*str == ':') {
        ++str;
        str = Common::Atof(str, &val);
        out_features->emplace_back(idx, val);
      } else {
        Log::Fatal("Input format error when parsing as LibSVM");
      }
      str = Common::SkipSpaceAndTab(str);
    }
  }

  inline int NumFeatures() const override {
    return total_columns_;
  }

 private:
  int label_idx_ = 0;
  int total_columns_ = -1;
  AtofFunc atof_;
};

class TransformParser: public Parser {
 public:
  explicit TransformParser(AtofFunc atof, string transform_file, string input_head_path)
    :atof_(atof), transform_file_(transform_file){
    Logger::Info << "Start initializing transform." << endl;
    transform_ = new TransformProcessor(transform_file, input_head_path,
                                        "Label0:0|Label1:1|Label2:2|Label3:3|Label4:4|Label5:5|Label6:6|Label7:7|Label8:8|Label9:9|Label10:10|Label11:11|Label12:12|Label13:13|Label14:14|Label15:15|Label16:16|Label17:17|Label18:18|Label19:19|Label20:20|Label21:21|Label22:22|Label23:23|Label24:24|Label25:25|Label26:26|Label27:27|Label28:28|Label29:29|Label30:30|Label31:31|Label32:32|Label33:33|Label34:34|Label35:35|Label36:36|Label37:37|Label38:38|Label39:39|Label40:40|Label41:41|Label42:42|Label43:43|Label44:44|Label45:45|Label46:46|Label47:47|Label48:48|Label49:49|Label50:50|Label51:51|Label52:52|Label53:53|Label54:54|Label55:55|Label56:56|Label57:57|Label58:58|Label59:59|Label60:60|Label61:61|Label62:62|Label63:63|Label64:64|Label65:65|Label66:66|Label67:67|Label68:68|Label69:69|Label70:70|Label71:71|Label72:72|Label73:73|Label74:74|Label75:75|Label76:76|Label77:77|Label78:78|Label79:79|Label80:80|Label81:81|Label82:82|Label83:83|Label84:84|Label85:85|Label86:86|Label87:87|Label88:88|Label89:89|Label90:90|Label91:91|Label92:92|Label93:93|Label94:94|Label95:95|Label96:96|Label97:97|Label98:98|Label99:99|Label100:100|Label101:101|Label102:102|Label103:103|Label104:104|Label105:105|Label106:106|Label107:107|Label108:108|Label109:109|Label110:110|Label111:111|Label112:112|Label113:113|Label114:114|Label115:115|Label116:116|Label117:117|Label118:118|Label119:119|Label120:120|Label121:121|Label122:122|Label123:123|Label124:124|Label125:125|Label126:126|Label127:127|Label128:128|Label129:129|Label130:130|Label131:131|Label132:132|Label133:133|Label134:134|Label135:135|Label136:136|Label137:137|Label138:138|Label139:139|Label140:140|Label141:141|Label142:142|Label143:143|Label144:144|Label145:145|Label146:146|Label147:147|Label148:148|Label149:149|Label150:150|Label151:151|Label152:152|Label153:153|Label154:154|Label155:155|Label156:156|Label157:157|Label158:158|Label159:159|Label160:160|Label161:161|Label162:162|Label163:163|Label164:164|Label165:165|Label166:166|Label167:167|Label168:168|Label169:169|Label170:170|Label171:171|Label172:172|Label173:173|Label174:174|Label175:175|Label176:176|Label177:177|Label178:178|Label179:179|Label180:180|Label181:181|Label182:182|Label183:183|Label184:184|Label185:185|Label186:186|Label187:187|Label188:188|Label189:189|Label190:190|Label191:191|Label192:192|Label193:193|Label194:194|Label195:195|Label196:196|Label197:197|Label198:198|Label199:199|Label200:200");
    Logger::Info << "Done initializing transform." << endl;
  }
  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features, double* out_label) const override {
    //TODO: make a function in transform lib for below codes.
    string sstr(str);
    TransformedData data = transform_->Apply(sstr);
    //TODO: label could be string?
    double label_val = 0.0f;
    atof_(data.Label().c_str(), &label_val);
    *out_label = label_val;
    for(auto f: data.Features()) {
      out_features->emplace_back(f.first, f.second);
    }
  }

  inline int NumFeatures() const override {
    return transform_->GetFeatureCount();
  }

 private:
  AtofFunc atof_;
  string transform_file_;
  TransformProcessor* transform_;
};

}  // namespace LightGBM
#endif   // LightGBM_IO_PARSER_HPP_
