#include <cfloat>
#include <vector>

#include "caffe/layers/rn_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void RNDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK(this->layer_param_.rn_dropout_param().num_drop_blobs() < bottom.size()) <<
			"num_drop_blobs should be less than the number of bottom blobs.";
		num_drop_blobs_ = this->layer_param_.rn_dropout_param().num_drop_blobs();
		drop_thres_ = this->layer_param_.rn_dropout_param().drop_prob();
		num_drop_combis_ = 0;
		if (this->layer_param_.rn_dropout_param().drop_seqs().size() > 0){
			num_drop_combis_ = this->layer_param_.rn_dropout_param().drop_seqs().size() / num_drop_blobs_;
			for (unsigned int i = 0; i < num_drop_combis_; i++){
				vector<unsigned int> tempVec;
				for (unsigned int j = 0; j < num_drop_blobs_; j++){
					tempVec.push_back(this->layer_param_.rn_dropout_param().drop_seqs(num_drop_blobs_*i + j));
				}
				drop_seqs_.push_back(tempVec);
			}
		}
		LOG(INFO) << "drop combinations : " << num_drop_combis_ << ", drop blobs : " << num_drop_blobs_;

		//scale init
		for (unsigned int i = 0; i < bottom[0]->num(); i++){
			scale_.push_back(Dtype(0));
		}
	}

	template <typename Dtype>
	void RNDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		for (int i = 0; i < bottom.size(); ++i) {
			top[i]->ReshapeLike(*bottom[i]);
		}
		rand_vec_.Reshape(bottom[0]->num()*bottom.size(), 1, 1, 1);
	}

	template <typename Dtype>
	void RNDropoutLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LOG(ERROR) << "CPU implementation not available for RNDropout layer.";
	}

	template <typename Dtype>
	void RNDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		LOG(ERROR) << "CPU implementation not available for RNDropout layer.";
	}

#ifdef CPU_ONLY
	STUB_GPU(RNDropoutLayer);
#endif

	INSTANTIATE_CLASS(RNDropoutLayer);
	REGISTER_LAYER_CLASS(RNDropout);

}  // namespace caffe
