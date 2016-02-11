#ifndef PYTHON_CAFFE__CAFFE_HPP_
#define PYTHON_CAFFE__CAFFE_HPP_

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <numpy/arrayobject.h>

// these need to be included after boost on OS X
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)

#include "caffe/caffe.hpp"

namespace bp = boost::python;
using boost::shared_ptr;

namespace caffe {

class PyNet {
 public:
  // For cases where parameters will be determined later by the Python user,
  // create a Net with unallocated parameters (which will not be zero-filled
  // when accessed).
  explicit PyNet(string param_file, int phase) { Init(param_file, static_cast<Phase>(phase)); }
  PyNet(string param_file, string pretrained_param_file, int phase);
  explicit PyNet(shared_ptr<Net<float> > net)
      : net_(net) {}
  virtual ~PyNet() {}

  void Init(string param_file, int phase);


  // Generate Python exceptions for badly shaped or discontiguous arrays.
  inline void check_contiguous_array(PyArrayObject* arr, string name,
      int channels, int height, int width);

  void Forward(int start, int end) { net_->ForwardFromTo(start, end); }
  void Backward(int start, int end) { net_->BackwardFromTo(start, end); }
  void Reshape() { net_->Reshape(); }

  void set_input_arrays(bp::object data_obj, bp::object labels_obj);
  // Accepts network parameters (weights) as a list of numpy arrays
  void set_params_data(bp::list data_obj);
  void set_params_diff(bp::list data_obj);

  // Save the network weights to binary proto for net surgeries.
  void save(string filename) {
    NetParameter net_param;
    net_->ToProto(&net_param, false);
    WriteProtoToBinaryFile(net_param, filename.c_str());
  }

  vector<shared_ptr<Blob<float> > > blobs() {
    return vector<shared_ptr<Blob<float> > >(net_->blobs().begin(), net_->blobs().end());
  }

  vector<shared_ptr<Layer<float> > > layers() {
    return vector<shared_ptr<Layer<float> > >(net_->layers().begin(), net_->layers().end());
  }

  vector<string> blob_names() { return net_->blob_names(); }
  vector<string> layer_names() { return net_->layer_names(); }

  bp::list inputs() {
    bp::list input_blob_indices;
    for (int i = 0; i < net_->input_blob_indices().size(); ++i) {
      input_blob_indices.append(net_->input_blob_indices()[i]);
    }
    return input_blob_indices;
  }

  bp::list outputs() {
    bp::list output_blob_indices;
    for (int i = 0; i < net_->output_blob_indices().size(); ++i) {
      output_blob_indices.append(net_->output_blob_indices()[i]);
    }
    return output_blob_indices;
  }

  // Returns all net parameters as list of numpy arrays
  bp::list params_data();
  bp::list params_diff();

  // Input preprocessing configuration attributes. These are public for
  // direct access from Python.
  bp::dict mean_;
  bp::dict input_scale_;
  bp::dict raw_scale_;
  bp::dict channel_swap_;

 protected:
  // The pointer to the internal caffe::Net instant.
  shared_ptr<Net<float> > net_;
  // if taking input from an ndarray, we need to hold references
  bp::object input_data_;
  bp::object input_labels_;
  bp::list params_data_;
};

template <typename SolverType>
class StepSolver : public SolverType
{
public:
  StepSolver(SolverParameter solver_param):SolverType(solver_param) {}
  StepSolver(const string& param_file):SolverType(param_file) {}
  virtual ~StepSolver(){}
  void InitSolve(string resume_file);
  void forward_backward();
  void update() { this->net_->Update(); }
  void compute_update_value() {  return this->ComputeUpdateValue(); }
  //void test() { this->TestAll(); }

  void calculate_train_info();
  void output_train_info();
  void output_train_loss();
  void output_learning_rate();
  void output_finish();
  
  void run_test();

  void clear_history();
  void snapshot() { this->Snapshot(); }

  int get_iter() const {return this->iter_; }
  void set_iter(int i) {this->iter_ = i; }

  float get_train_loss() const { return this->train_loss_; }
  void set_train_loss(float train_loss) { this->train_loss_ = train_loss; }

  float get_train_accuracy() const { return this->train_accuracy_; }
  void set_train_accuracy(float train_accuracy) { this->train_accuracy_ = train_accuracy; }

  string get_train_info() const { return this->train_info_; }
  void set_train_info(const string& train_info) { this->train_info_ = train_info; }

  float get_val_loss() const { return this->val_loss_; }
  void set_val_loss(float val_loss) { this->val_loss_ = val_loss; }

  float get_val_accuracy() const { return this->val_accuracy_; }
  void set_val_accuracy(float val_accuracy) { this->val_accuracy_ = val_accuracy; }


  float get_learning_rate() {return this->GetLearningRate(); }
  int max_iter() const { return this->param_.max_iter(); }
  int test_interval() const { return this->param_.test_interval(); }
  int snapshot_interval() const { return this->param_.snapshot() ; }
  int display() const { return this->param_.display(); }

private:
  float train_loss_;
  float train_accuracy_;
  float val_loss_;
  float val_accuracy_;

  string train_info_;

};

template <typename StepSolverType>
class PySolver {
public:
  explicit PySolver(const string& param_file);

  shared_ptr<PyNet> net() { return this->net_; }
  void Solve() { return this->solver_->Solve(); }
  void SolveResume(const string& resume_file);
  void Snapshot() { this->solver_->snapshot(); }

  void InitSolve(string resume_file) { this->solver_->InitSolve(resume_file); }
  void forward_backward() { this->solver_->forward_backward(); }
  void update() { this->solver_->update(); }
  void compute_update_value() { this->solver_->compute_update_value(); }
  void clear_history() { this->solver_->clear_history(); }
  void run_test() { this->solver_->run_test(); }

  void calculate_train_info() { this->solver_->calculate_train_info(); }
  void output_train_info() { this->solver_->output_train_info(); }
  void output_train_loss() { this->solver_->output_train_loss(); }
  void output_learning_rate() { this->solver_->output_learning_rate(); }
  void output_finish() { this->solver_->output_finish(); }

  int get_iter() const {return this->solver_->get_iter(); }
  void set_iter(int i) {this->solver_->set_iter(i); }

  float get_train_loss() const { return this->solver_->get_train_loss(); }
  void set_train_loss(float train_loss) { this->solver_->set_train_loss(train_loss); }
  
  float get_train_accuracy() const { return this->solver_->get_train_accuracy(); }
  void set_train_accuracy(float train_accuracy) { this->solver_->set_train_accuracy(train_accuracy); }

  float get_val_loss() const { return this->solver_->get_val_loss(); }
  void set_val_loss(float val_loss) { this->solver_->set_val_loss(val_loss); }

  float get_val_accuracy() const { return this->solver_->get_val_accuracy(); }
  void set_val_accuracy(float val_accuracy) { this->solver_->set_val_accuracy(val_accuracy); }
  
  string get_train_info() { return this->solver_->get_train_info(); }
  void set_train_info(const string& train_info) { this->solver_->set_train_info(train_info); }

  int max_iter() const { return this->solver_->max_iter(); }
  float train_loss() const { return this->solver_->train_loss(); }

  int test_interval() const { return this->solver_->test_interval(); }
  int snapshot_interval() const { return this->solver_->snapshot_interval() ; }
  int display() const { return this->solver_->display(); }

  float get_learning_rate() {return this->solver_->get_learning_rate();}
protected:
  shared_ptr<PyNet> net_;
  shared_ptr<StepSolverType> solver_;
};

typedef PySolver<StepSolver<SGDSolver<float> > > PySGDSolver;
typedef PySolver<StepSolver<NesterovSolver<float> > > PyNesterovSolver;
typedef PySolver<StepSolver<AdaGradSolver<float> > > PyAdaGradSolver;

// Declare the module init function created by boost::python, so that we can
// use this module from C++ when embedding Python.
PyMODINIT_FUNC init_caffe_facade(void);

}  // namespace caffe

#endif
