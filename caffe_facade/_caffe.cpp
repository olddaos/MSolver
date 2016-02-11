// pycaffe provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from Python.
// Note that for Python, we will simply use float as the data type.
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

// these need to be included after boost on OS X
#include <string>  // NOLINT(build/include_order)
#include <sstream>
#include <vector>  // NOLINT(build/include_order)
#include <fstream>  // NOLINT
#include <cstdio>

#include "_caffe.hpp"
#include "caffe/caffe.hpp"
#include "caffe/python_layer.hpp"

// Temporary solution for numpy < 1.7 versions: old macro, no promises.
// You're strongly advised to upgrade to >= 1.7.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define PyArray_SetBaseObject(arr, x) (PyArray_BASE(arr) = (x))
#endif

namespace caffe {
typedef float Dtype;
const int NPY_DTYPE = NPY_FLOAT32;

// Selecting mode.
void set_mode_cpu() { Caffe::set_mode(Caffe::CPU); }
void set_mode_gpu() { Caffe::set_mode(Caffe::GPU); }
// Checking current mode.
bool check_mode_cpu() { return Caffe::mode() == Caffe::CPU; }
bool check_mode_gpu() { return Caffe::mode() == Caffe::GPU; }
#ifndef CPU_ONLY
// Cuda num threads
int get_cuda_num_threads() { return CAFFE_CUDA_NUM_THREADS; }
bp::object cublas_handle() {
  return bp::object((size_t)Caffe::cublas_handle());
}
#endif

// for convenience, check that input files can be opened, and raise an
// exception that boost will send to Python if not (caffe could still crash
// later if the input files are disturbed before they are actually used, but
// this saves frustration in most cases)
static void CheckFile(const string& filename) {
    std::ifstream f(filename.c_str());
    if (!f.good()) {
      f.close();
      throw std::runtime_error("Could not open file " + filename);
    }
    f.close();
}

PyNet::PyNet(string param_file, string pretrained_param_file, int phase) {
  Init(param_file, phase);
  CheckFile(pretrained_param_file);
  net_->CopyTrainedLayersFrom(pretrained_param_file);
}

void PyNet::Init(string param_file, int phase) {
  CheckFile(param_file);
  net_.reset(new Net<float>(param_file, static_cast<Phase>(phase)));
}

void PyNet::check_contiguous_array(PyArrayObject* arr, string name,
    int channels, int height, int width) {
  if (!(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS)) {
    throw std::runtime_error(name + " must be C contiguous");
  }
  if (PyArray_NDIM(arr) != 4) {
    throw std::runtime_error(name + " must be 4-d");
  }
  if (PyArray_TYPE(arr) != NPY_FLOAT32) {
    throw std::runtime_error(name + " must be float32");
  }
  if (PyArray_DIMS(arr)[1] != channels) {
    throw std::runtime_error(name + " has wrong number of channels");
  }
  if (PyArray_DIMS(arr)[2] != height) {
    throw std::runtime_error(name + " has wrong height");
  }
  if (PyArray_DIMS(arr)[3] != width) {
    throw std::runtime_error(name + " has wrong width");
  }
}

void PyNet::set_input_arrays(bp::object data_obj, bp::object labels_obj) {
  // check that this network has an input MemoryDataLayer
  shared_ptr<MemoryDataLayer<float> > md_layer =
    boost::dynamic_pointer_cast<MemoryDataLayer<float> >(net_->layers()[0]);
  if (!md_layer) {
    throw std::runtime_error("set_input_arrays may only be called if the"
        " first layer is a MemoryDataLayer");
  }

  // check that we were passed appropriately-sized contiguous memory
  PyArrayObject* data_arr =
      reinterpret_cast<PyArrayObject*>(data_obj.ptr());
  PyArrayObject* labels_arr =
      reinterpret_cast<PyArrayObject*>(labels_obj.ptr());
  check_contiguous_array(data_arr, "data array", md_layer->channels(),
      md_layer->height(), md_layer->width());
  check_contiguous_array(labels_arr, "labels array", 1, 1, 1);
  if (PyArray_DIMS(data_arr)[0] != PyArray_DIMS(labels_arr)[0]) {
    throw std::runtime_error("data and labels must have the same first"
        " dimension");
  }
  if (PyArray_DIMS(data_arr)[0] % md_layer->batch_size() != 0) {
    throw std::runtime_error("first dimensions of input arrays must be a"
        " multiple of batch size");
  }

  // hold references
  input_data_ = data_obj;
  input_labels_ = labels_obj;

  md_layer->Reset(static_cast<float*>(PyArray_DATA(data_arr)),
      static_cast<float*>(PyArray_DATA(labels_arr)),
      PyArray_DIMS(data_arr)[0]);
}

void PyNet::set_params_data(bp::list params_list) {
  // Check number of layers
  if(bp::len(params_list) != this->net_->params().size()) {
    std::runtime_error("params list size should be equal to number of layers");
  }
  // hold references
  params_data_ = params_list;
  for(int i = 0; i < bp::len(params_list); ++i){
    // check that we were passed appropriately-sized contiguous memory
    PyArrayObject* data_arr =
      reinterpret_cast<PyArrayObject*>(bp::object(params_list[i]).ptr());
    shared_ptr< Blob<float> > layer_param = net_->params()[i];
    check_contiguous_array(data_arr, "data array", layer_param->channels(),
        layer_param->height(), layer_param->width());
    // For params all dimensions should match
    if (PyArray_DIMS(data_arr)[0] != layer_param->num()) {
      stringstream ss;
      ss<<"Layer "<<i<<" params has wrong num";
      throw std::runtime_error(ss.str());
    }
    layer_param->set_cpu_data(static_cast<float*>(PyArray_DATA(data_arr)));
  }
}

// Despite set_params_data, this function copies values from input array
// instead of sharing the buffer. This is due to absense of set_cpu_diff method in blob
void PyNet::set_params_diff(bp::list params_list) {
  // Check number of layers
  if(bp::len(params_list) != this->net_->params().size()) {
    std::runtime_error("params list size should be equal to number of layers");
  }
  for(int i = 0; i < bp::len(params_list); ++i){
    // check that we were passed appropriately-sized contiguous memory
    PyArrayObject* data_arr =
      reinterpret_cast<PyArrayObject*>(bp::object(params_list[i]).ptr());
    shared_ptr< Blob<float> > layer_param = net_->params()[i];
    check_contiguous_array(data_arr, "data array", layer_param->channels(),
        layer_param->height(), layer_param->width());
    // For params all dimensions should match
    if (PyArray_DIMS(data_arr)[0] != layer_param->num()) {
      stringstream ss;
      ss<<"Layer "<<i<<" params has wrong num";
      throw std::runtime_error(ss.str());
    }
    // No set_cpu_diff available
    // layer_param->set_cpu_diff(static_cast<float*>(PyArray_DATA(data_arr)));
    memcpy(layer_param->mutable_cpu_diff(), PyArray_DATA(data_arr),
        layer_param->count()*sizeof(float));
  }
}

bp::list PyNet::params_data() {
  bp::list params_list;
  for(int i = 0;  i < net_->params().size(); ++i) {
    shared_ptr< Blob<float> > layer_param = net_->params()[i];
    npy_intp dims[] = {layer_param->num(), layer_param->channels(), layer_param->height(),
      layer_param->width()};
    PyObject *obj = PyArray_SimpleNewFromData(4, dims, NPY_FLOAT32,
        layer_param->mutable_cpu_data());
    bp::handle<> h(obj);
    params_list.append(bp::object(h));
  }
  return params_list;
}

bp::list PyNet::params_diff() {
  bp::list params_list;
  for(int i = 0;  i < net_->params().size(); ++i) {
    shared_ptr< Blob<float> > layer_param = net_->params()[i];
    npy_intp dims[] = {layer_param->num(), layer_param->channels(), layer_param->height(),
      layer_param->width()};
    PyObject *obj = PyArray_SimpleNewFromData(4, dims, NPY_FLOAT32,
        layer_param->mutable_cpu_diff());
    bp::handle<> h(obj);
    params_list.append(bp::object(h));
  }
  return params_list;
}

template <typename SolverType>
void StepSolver<SolverType>::InitSolve(string resume_file) {
  ::google::InitGoogleLogging("");
  ::google::SetLogDestination(google::INFO, "./CAFFE_UNIT_LOG_"); 
  Caffe::set_mode(Caffe::Brew(this->param_.solver_mode()));
  if (this->param_.solver_mode() && this->param_.has_device_id()) {
    Caffe::SetDevice(this->param_.device_id());
  }
  
  LOG(INFO) << "Solving " << this->net_->name();
  LOG(INFO) << "Learning Rate Policy: " << this->param_.lr_policy();
  this->PreSolve();

  this->iter_ = 0;
  if (resume_file.size()) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    this->Restore(resume_file.c_str());
  }
}

template <typename SolverType>
void StepSolver<SolverType>::forward_backward(){
  vector<Blob<float>*> bottom_vec;
  this->train_loss_ = this->net_->ForwardBackward(bottom_vec);
}

template <typename SolverType>
void StepSolver<SolverType>::clear_history(){
  const vector<shared_ptr<Blob<float> > >& net_params = this->net_->params();
  this->history_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const Blob<float>* net_param = net_params[i].get();
    this->history_.push_back(shared_ptr<Blob<float> >(new Blob<float>(
    net_param->num(), net_param->channels(), net_param->height(),
    net_param->width())));
  }
}

template <typename SolverType>
void StepSolver<SolverType>::calculate_train_info() {
  ostringstream train_info;

  const vector<Blob<Dtype>*>& result = this->net_->output_blobs();
  int score_index = 0;
  for (int j = 0; j < result.size(); ++j) {
    const float* result_vec = result[j]->cpu_data();
    const string& output_name =
    this->net_->blob_names()[this->net_->output_blob_indices()[j]];
    const float loss_weight =
    this->net_->blob_loss_weights()[this->net_->output_blob_indices()[j]];
    for (int k = 0; k < result[j]->count(); ++k) {
      ostringstream loss_msg_stream;
      if (loss_weight) {
        loss_msg_stream << " (* " << loss_weight
        << " = " << loss_weight * result_vec[k] << " loss)";
      }
      train_info << "    Train net output #"
          << score_index++ << ": " << output_name << " = "
          << result_vec[k] << loss_msg_stream.str() << std::endl;

      if (output_name == "accuracy" || output_name == "Accuracy") {
        this->train_accuracy_ = result_vec[k];
      }
    }
  }
  this->train_info_ = train_info.str();
}

template <typename SolverType>
void StepSolver<SolverType>::output_train_info() {
  std::istringstream train_info(this->train_info_);
  while (!train_info.eof()) {
    string out;
    getline(train_info, out);
    if (!out.empty()) {
      LOG(INFO) << out;
    }
  }
}

template <typename SolverType>
void StepSolver<SolverType>::run_test() {
  const int test_net_id = 0;
  LOG(INFO) << "Iteration " << this->iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(this->test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(this->net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  vector<Blob<Dtype>*> bottom_vec;
  const shared_ptr<Net<Dtype> >& test_net = this->test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < this->param_.test_iter(test_net_id); ++i) {
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(bottom_vec, &iter_loss);
    if (this->param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (this->param_.test_compute_loss()) {
    loss /= this->param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
    this->val_loss_ = loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / this->param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
        << mean_score << loss_msg_stream.str();
    if (output_name == "loss" || output_name == "Loss") {
        LOG(INFO) << "Test loss: " << mean_score;
        this->val_loss_ = mean_score;
      }
    if (output_name == "accuracy" || output_name == "Accuracy") {
        LOG(INFO) << "Test accuracy: " << mean_score;
        this->val_accuracy_ = mean_score;
      }
  }
}


template <typename SolverType>
void StepSolver<SolverType>::output_train_loss() {
  LOG(INFO) << "Iteration " << this->iter_ << ", loss = " << this->train_loss_;
}

template <typename SolverType>
void StepSolver<SolverType>::output_learning_rate() {
  LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << this->get_learning_rate();
}

template <typename SolverType>
void StepSolver<SolverType>::output_finish() {
  LOG(INFO) << "Optimization Done.";
}

template <typename SolverType>
PySolver<SolverType>::PySolver(const string& param_file) {
  // as in PyNet, (as a convenience, not a guarantee), create a Python
  // exception if param_file can't be opened
  CheckFile(param_file);
  this->solver_.reset(new SolverType(param_file));
  // we need to explicitly store the net wrapper, rather than constructing
  // it on the fly, so that it can hold references to Python objects
  this->net_.reset(new PyNet(this->solver_->net()));
}

template <typename SolverType>
void PySolver<SolverType>::SolveResume(const string& resume_file) {
  CheckFile(resume_file);
  return this->solver_->Solve(resume_file);
}

struct NdarrayConverterGenerator {
  template <typename T> struct apply;
};

template <>
struct NdarrayConverterGenerator::apply<Dtype*> {
  struct type {
    PyObject* operator() (Dtype* data) const {
      // Just store the data pointer, and add the shape information in postcall.
      return PyArray_SimpleNewFromData(0, NULL, NPY_DTYPE, data);
    }
    const PyTypeObject* get_pytype() {
      return &PyArray_Type;
    }
  };
};

struct NdarrayCallPolicies : public bp::default_call_policies {
  typedef NdarrayConverterGenerator result_converter;
  PyObject* postcall(PyObject* pyargs, PyObject* result) {
    bp::object pyblob = bp::extract<bp::tuple>(pyargs)()[0];
    shared_ptr<Blob<Dtype> > blob =
    bp::extract<shared_ptr<Blob<Dtype> > >(pyblob);
    // Free the temporary pointer-holding array, and construct a new one with
    // the shape information from the blob.
    void* data = PyArray_DATA(reinterpret_cast<PyArrayObject*>(result));
    Py_DECREF(result);
    const int num_axes = blob->num_axes();
    vector<npy_intp> dims(blob->shape().begin(), blob->shape().end());
    PyObject *arr_obj = PyArray_SimpleNewFromData(num_axes, dims.data(),
    NPY_FLOAT32, data);
    // SetBaseObject steals a ref, so we need to INCREF.
    Py_INCREF(pyblob.ptr());
    PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(arr_obj),
    pyblob.ptr());
    return arr_obj;
  }
};

// Blob constructor with shape iterable
shared_ptr<Blob<Dtype> > Blob_Init(bp::object shape_object) {
  size_t ndim;
  try {
    ndim = bp::len(shape_object);
  } catch(...) {
    throw std::runtime_error("1st arg must be iterable.");
  }
  vector<int> shape(ndim);
  try {
    for (int i = 0; i < ndim; ++i) {
      shape[i] = bp::extract<int>(shape_object[i]);
    }
  } catch(...) {
    throw std::runtime_error("All element in shape iterable must be integer.");
  }
  return shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape));
}

bp::object Blob_Shape(bp::tuple args, bp::dict kwargs) {
  if (bp::len(kwargs) > 0) {
    throw std::runtime_error("Blob.shape takes no kwargs");
  }
  Blob<Dtype>* self = bp::extract<Blob<Dtype>*>(args[0]);
  const vector<int> &shape = self->shape();
  bp::list shape_list;
  BOOST_FOREACH(int s, shape) {
    shape_list.append(s);
  }
  return bp::tuple(shape_list);
}

bp::object Blob_Reshape(bp::tuple args, bp::dict kwargs) {
  if (bp::len(kwargs) > 0) {
    throw std::runtime_error("Blob.reshape takes no kwargs");
  }
  Blob<Dtype>* self = bp::extract<Blob<Dtype>*>(args[0]);
  vector<int> shape(bp::len(args) - 1);
  for (int i = 1; i < bp::len(args); ++i) {
    shape[i - 1] = bp::extract<int>(args[i]);
  }
  self->Reshape(shape);
  // We need to explicitly return None to use bp::raw_function.
  return bp::object();
}

#ifndef CPU_ONLY
bp::object Blob_GpuDataPtr(bp::tuple args, bp::dict kwargs) {
  if (bp::len(kwargs) > 0) {
    throw std::runtime_error("Blob.gpu_data_ptr takes no kwargs");
  }
  Blob<Dtype>* self = bp::extract<Blob<Dtype>*>(args[0]);
  return bp::object((size_t)(self->mutable_gpu_data()));
}

bp::object Blob_GpuDiffPtr(bp::tuple args, bp::dict kwargs) {
  if (bp::len(kwargs) > 0) {
    throw std::runtime_error("Blob.gpu_diff_ptr takes no kwargs");
  }
  Blob<Dtype>* self = bp::extract<Blob<Dtype>*>(args[0]);
  return bp::object((size_t)(self->mutable_gpu_diff()));
}
#endif

BOOST_PYTHON_MODULE(_caffe_facade) {
  // below, we prepend an underscore to methods that will be replaced
  // in Python
  // Caffe utility functions
  bp::def("set_mode_cpu", &set_mode_cpu);
  bp::def("set_mode_gpu", &set_mode_gpu);
  bp::def("check_mode_cpu", &check_mode_cpu);
  bp::def("check_mode_gpu", &check_mode_gpu);
  bp::def("set_device", &Caffe::SetDevice);
  #ifndef CPU_ONLY
  bp::def("get_cuda_num_threads", &get_cuda_num_threads);
  bp::def("get_blocks", &CAFFE_GET_BLOCKS);
  bp::def("cublas_handle", &cublas_handle);
  #endif

  bp::class_<Blob<Dtype>, shared_ptr<Blob<Dtype> >, boost::noncopyable>(
    "Blob", bp::no_init)
    .def("__init__", bp::make_constructor(&Blob_Init))
    .add_property("num",      &Blob<Dtype>::num)
    .add_property("channels", &Blob<Dtype>::channels)
    .add_property("height",   &Blob<Dtype>::height)
    .add_property("width",    &Blob<Dtype>::width)
    .add_property("count",    static_cast<int (Blob<Dtype>::*)() const>(
        &Blob<Dtype>::count))
    .add_property("shape", bp::raw_function(&Blob_Shape))
    .def("reshape",           bp::raw_function(&Blob_Reshape))
  #ifndef CPU_ONLY
    .add_property("gpu_data_ptr", bp::raw_function(&Blob_GpuDataPtr))
    .add_property("gpu_diff_ptr", bp::raw_function(&Blob_GpuDiffPtr))
  #endif
    .add_property("data",     bp::make_function(&Blob<Dtype>::mutable_cpu_data,
        NdarrayCallPolicies()))
    .add_property("diff",     bp::make_function(&Blob<Dtype>::mutable_cpu_diff,
        NdarrayCallPolicies()));

  bp::class_<Layer<Dtype>, shared_ptr<PythonLayer<Dtype> >,
    boost::noncopyable>("Layer", bp::init<const LayerParameter&>())
    .add_property("blobs", bp::make_function(&Layer<Dtype>::blobs,
        bp::return_internal_reference<>()))
    .def("setup", &Layer<Dtype>::LayerSetUp)
    .def("reshape", &Layer<Dtype>::Reshape)
    .add_property("type", bp::make_function(&Layer<Dtype>::type));
        bp::register_ptr_to_python<shared_ptr<Layer<Dtype> > >();

  // below, we prepend an underscore to methods that will be replaced
  // in Python
  bp::class_<PyNet, shared_ptr<PyNet> >(
      "Net", bp::init<string, string, int>())
      .def(bp::init<string, int>())
      .def("_forward",              &PyNet::Forward)
      .def("_backward",             &PyNet::Backward)
      .def("reshape",               &PyNet::Reshape)
      /*
      .add_property("_blobs",       bp::make_function(&Net<float>::blobs,
          bp::return_internal_reference<>()))
      .add_property("layers",       bp::make_function(&Net<float>::layers,
          bp::return_internal_reference<>()))
      .add_property("_blob_names", bp::make_function(&Net<float>::blob_names,
          bp::return_value_policy<bp::copy_const_reference>()))
      .add_property("_layer_names", bp::make_function(&Net<float>::layer_names,
          bp::return_value_policy<bp::copy_const_reference>()))
      */
      .add_property("_blobs",       &PyNet::blobs)
      .add_property("layers",       &PyNet::layers)
      .add_property("_blob_names",  &PyNet::blob_names)
      .add_property("_layer_names", &PyNet::layer_names)
      .add_property("_inputs",      &PyNet::inputs)
      .add_property("_outputs",     &PyNet::outputs)
      .add_property("mean",         &PyNet::mean_)
      .add_property("input_scale",  &PyNet::input_scale_)
      .add_property("raw_scale",    &PyNet::raw_scale_)
      .add_property("channel_swap", &PyNet::channel_swap_)
      .add_property("params_data",  &PyNet::params_data, &PyNet::
          set_params_data)
      .add_property("params_diff",  &PyNet::params_diff, &PyNet::
          set_params_diff)
      .def("_set_input_arrays",     &PyNet::set_input_arrays)
      .def("save",                  &PyNet::save);


  bp::class_<PySGDSolver, boost::noncopyable>(
      "SGDSolver", bp::init<string>())
      .add_property("net",                   &PySGDSolver::net)
      .def("solve",                          &PySGDSolver::Solve)
      .def("solve",                          &PySGDSolver::SolveResume)
      .def("init_solve",                     &PySGDSolver::InitSolve)
      .def("forward_backward",               &PySGDSolver::forward_backward)
      .def("update",                         &PySGDSolver::update)
      .def("compute_update_value",           &PySGDSolver::compute_update_value)
      .def("run_test",                           &PySGDSolver::run_test)
      .def("calculate_train_info",           &PySGDSolver::calculate_train_info)
      .def("output_train_info",              &PySGDSolver::output_train_info)
      .def("output_train_loss",              &PySGDSolver::output_train_loss)
      .def("output_learning_rate",           &PySGDSolver::output_learning_rate)
      .def("output_finish",                  &PySGDSolver::output_finish)
      .def("snapshot",                       &PySGDSolver::Snapshot)
      .def("get_learning_rate",              &PySGDSolver::get_learning_rate)
      .def("clear_history",                  &PySGDSolver::clear_history)
      .add_property("iter",                  &PySGDSolver::get_iter, &PySGDSolver::set_iter)
      .add_property("max_iter",              &PySGDSolver::max_iter)
      .add_property("test_interval",         &PySGDSolver::test_interval)
      .add_property("snapshot_interval",     &PySGDSolver::snapshot_interval)
      .add_property("display",               &PySGDSolver::display)
      .add_property("train_info",            &PySGDSolver::get_train_info, &PySGDSolver::set_train_info)
      .add_property("train_loss",            &PySGDSolver::get_train_loss, &PySGDSolver::set_train_loss)
       .add_property("train_accuracy",        &PySGDSolver::get_train_accuracy, &PySGDSolver::set_train_accuracy)
      .add_property("val_loss",              &PySGDSolver::get_val_loss, &PySGDSolver::set_val_loss)
      .add_property("val_accuracy",          &PySGDSolver::get_val_accuracy, &PySGDSolver::set_val_accuracy);

  bp::class_<PyNesterovSolver, boost::noncopyable>(
      "NesterovSolver", bp::init<string>())
      .add_property("net",                   &PyNesterovSolver::net)
      .def("solve",                          &PyNesterovSolver::Solve)
      .def("solve",                          &PyNesterovSolver::SolveResume)
      .def("init_solve",                     &PyNesterovSolver::InitSolve)
      .def("forward_backward",               &PyNesterovSolver::forward_backward)
      .def("update",                         &PyNesterovSolver::update)
      .def("compute_update_value",           &PyNesterovSolver::compute_update_value)
      .def("run_test",                           &PyNesterovSolver::run_test)
      .def("calculate_train_info",           &PyNesterovSolver::calculate_train_info)
      .def("output_train_info",              &PyNesterovSolver::output_train_info)
      .def("output_train_loss",              &PyNesterovSolver::output_train_loss)
      .def("output_learning_rate",           &PyNesterovSolver::output_learning_rate)
      .def("output_finish",                  &PyNesterovSolver::output_finish)
      .def("snapshot",                       &PyNesterovSolver::Snapshot)
      .def("get_learning_rate",              &PyNesterovSolver::get_learning_rate)
      .def("clear_history",                  &PyNesterovSolver::clear_history)
      .add_property("iter",                  &PyNesterovSolver::get_iter, &PyNesterovSolver::set_iter)
      .add_property("max_iter",              &PyNesterovSolver::max_iter)
      .add_property("test_interval",         &PyNesterovSolver::test_interval)
      .add_property("snapshot_interval",     &PyNesterovSolver::snapshot_interval)
      .add_property("display",               &PyNesterovSolver::display)
      .add_property("train_info",            &PyNesterovSolver::get_train_info, &PyNesterovSolver::set_train_info)
      .add_property("train_loss",            &PyNesterovSolver::get_train_loss, &PyNesterovSolver::set_train_loss)
            .add_property("train_accuracy",        &PyNesterovSolver::get_train_accuracy, &PySGDSolver::set_train_accuracy)
      .add_property("val_loss",              &PyNesterovSolver::get_val_loss, &PyNesterovSolver::set_val_loss)
      .add_property("val_accuracy",          &PyNesterovSolver::get_val_accuracy, &PyNesterovSolver::set_val_accuracy);


  bp::class_<PyAdaGradSolver, boost::noncopyable>(
      "AdaGradSolver", bp::init<string>())
      .add_property("net",                   &PyAdaGradSolver::net)
      .def("solve",                          &PyAdaGradSolver::Solve)
      .def("solve",                          &PyAdaGradSolver::SolveResume)
      .def("init_solve",                     &PyAdaGradSolver::InitSolve)
      .def("forward_backward",               &PyAdaGradSolver::forward_backward)
      .def("update",                         &PyAdaGradSolver::update)
      .def("compute_update_value",           &PyAdaGradSolver::compute_update_value)
      .def("run_test",                           &PyAdaGradSolver::run_test)
      .def("calculate_train_info",           &PyAdaGradSolver::calculate_train_info)
      .def("output_train_info",              &PyAdaGradSolver::output_train_info)
      .def("output_train_loss",              &PyAdaGradSolver::output_train_loss)
      .def("output_learning_rate",           &PyAdaGradSolver::output_learning_rate)
      .def("output_finish",                  &PyAdaGradSolver::output_finish)
      .def("snapshot",                       &PyAdaGradSolver::Snapshot)
      .def("get_learning_rate",              &PyAdaGradSolver::get_learning_rate)
      .def("clear_history",                  &PyAdaGradSolver::clear_history)
      .add_property("iter",                  &PyAdaGradSolver::get_iter, &PyAdaGradSolver::set_iter)
      .add_property("max_iter",              &PyAdaGradSolver::max_iter)
      .add_property("test_interval",         &PyAdaGradSolver::test_interval)
      .add_property("snapshot_interval",     &PyAdaGradSolver::snapshot_interval)
      .add_property("display",               &PyAdaGradSolver::display)
      .add_property("train_info",            &PyAdaGradSolver::get_train_info, &PyAdaGradSolver::set_train_info)
      .add_property("train_loss",            &PyAdaGradSolver::get_train_loss, &PyAdaGradSolver::set_train_loss)
      .add_property("train_accuracy",        &PyAdaGradSolver::get_train_accuracy, &PySGDSolver::set_train_accuracy)
      .add_property("val_loss",              &PyAdaGradSolver::get_val_loss, &PyAdaGradSolver::set_val_loss)
      .add_property("val_accuracy",          &PyAdaGradSolver::get_val_accuracy, &PySGDSolver::set_val_accuracy);



  bp::class_<vector<shared_ptr<Blob<float> > > >("BlobVec")
      .def(bp::vector_indexing_suite<vector<shared_ptr<Blob<float> > >, true>());

  bp::class_<vector<shared_ptr<Layer<float> > > >("LayerVec")
      .def(bp::vector_indexing_suite<vector<shared_ptr<Layer<float> > >, true>());

  bp::class_<vector<string> >("StringVec")
      .def(bp::vector_indexing_suite<vector<string> >());

  import_array();
}

}  // namespace caffe
