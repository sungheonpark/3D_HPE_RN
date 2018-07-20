#include <cfloat>
#include <vector>

#include "caffe/layers/rn_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void RNDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		if (this->phase_ == TRAIN) {
			const Dtype *bottomDat;
			Dtype *topDat;
			//Determine mask
			const int count = top[0]->count();
			const int num = top[0]->num();
			const int chw = count / num;
			//LOG(INFO) << "chw : " << chw << " count : " << count << " num : " << num;
			//Drop if mask==1!! not 0!
			unsigned int* mask = rand_vec_.mutable_cpu_data();
			memset(mask, 0, sizeof(unsigned int) * rand_vec_.count());
			for (int i = 0; i < num; i++){
				// for drop_comp_num=1 only
				if (caffe_rng_rand() % 10 < drop_thres_){
					//drop
					scale_[i] = Dtype(bottom.size()) / Dtype((bottom.size() - num_drop_blobs_));
					unsigned int dropIndex = caffe_rng_rand() % num_drop_combis_;
					vector<unsigned int> droppingElems = drop_seqs_[dropIndex];
					//droppingElems should start from 0.
					for (int j = 0; j < num_drop_blobs_; j++){
						mask[num * droppingElems[j] + i] = 1;
						//LOG(INFO) << "num : " << i << ", elem : " << droppingElems[j];
					}
				}
				else{
					//no drop
					scale_[i] = Dtype(1);
				}
			}
			//Forward pass
			//copy or drop
			for (int i = 0; i < bottom.size(); i++){
				bottomDat = bottom[i]->gpu_data();
				topDat = top[i]->mutable_gpu_data();
				for (int j = 0; j < num; j++){
					if (mask[i*num + j]){
						//drop
						caffe_gpu_set(chw, Dtype(0), topDat + (chw*j));
					}
					else{
						//no drop. scale.
						caffe_gpu_scale(chw, scale_[j], bottomDat + (chw*j), topDat + (chw*j));
					}
				}
			}
		}
		else{
			if (drop_seqs_.size() == 0){
				//test. just copy if drop_seqs_ are not specified.
				const int count = top[0]->count();
				for (int i = 0; i < bottom.size(); ++i) {
					Dtype* top_data = top[i]->mutable_gpu_data();
					const Dtype* bottom_data = bottom[i]->gpu_data();
					caffe_copy(count, bottom_data, top_data);
				}
			}
			else{
				//NOTE : only first seqs will be used for testing.
				const int count = top[0]->count();
				const int num = top[0]->num();
				const int chw = count / num;
				const Dtype *bottomDat;
				Dtype *topDat;
				unsigned int* mask = rand_vec_.mutable_cpu_data();
				memset(mask, 0, sizeof(unsigned int) * rand_vec_.count());
				vector<unsigned int> droppingElems = drop_seqs_[0];
				for (int i = 0; i < num; i++){
					scale_[i] = Dtype(bottom.size()) / Dtype((bottom.size() - num_drop_blobs_));
					for (int j = 0; j < num_drop_blobs_; j++){
						mask[num * droppingElems[j] + i] = 1;
					}
				}
				//forward pass
				for (int i = 0; i < bottom.size(); i++){
					bottomDat = bottom[i]->gpu_data();
					topDat = top[i]->mutable_gpu_data();
					for (int j = 0; j < num; j++){
						if (mask[i*num + j]){
							//drop
							caffe_gpu_set(chw, Dtype(0), topDat + (chw*j));
						}
						else{
							//no drop. scale.
							caffe_gpu_scale(chw, scale_[j], bottomDat + (chw*j), topDat + (chw*j));
						}
					}
				}
			}
		}
	}


	template <typename Dtype>
	void RNDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const int count = top[0]->count();
		const int num = top[0]->num();
		const int chw = count / num;
		for (int i = 0; i < bottom.size(); ++i) {

			const Dtype* top_diff = top[i]->gpu_diff();
			Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();

			if (propagate_down[i]) {
				if (this->phase_ == TRAIN) {
					const unsigned int* mask = rand_vec_.cpu_data();
					for (int j = 0; j < num; j++){
						if (mask[i*num + j]){
							caffe_gpu_set(chw, Dtype(0), bottom_diff + (chw*j));
						}
						else{
							caffe_gpu_scale(chw, scale_[j], top_diff + (chw*j), bottom_diff + (chw*j));
						}
					}
				}
				else{
					caffe_copy(count, top_diff, bottom_diff);

				}
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(RNDropoutLayer);

}  // namespace caffe
