#ifndef CAFFE_RN_DROPOUT_LAYER_HPP_
#define CAFFE_RN_DROPOUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	/**
	* @brief selective dropout for relational modules. # of Top blobs are the same as # of the bottom.
	*        Selectively drops off the components for 
	* TODO(dox): thorough documentation for Forward, Backward, and proto params.
	*/
	template <typename Dtype>
	class RNDropoutLayer : public Layer<Dtype> {
	public:
		explicit RNDropoutLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "RNDropout"; }
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		vector<vector<unsigned int>> drop_seqs_;
		unsigned int num_drop_blobs_;
		unsigned int num_drop_combis_;
		unsigned int drop_thres_;

		/// used for Mask
		Blob<unsigned int> rand_vec_;
		/// scale is needed for each example
		vector<Dtype> scale_;
	};

}  // namespace caffe

#endif  // CAFFE_RN_DROPOUT_LAYER_HPP_
