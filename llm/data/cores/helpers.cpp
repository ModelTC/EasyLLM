/*
 coding=utf-8
 Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */


/* Helper methods for fast index mapping builds */

#include <algorithm>
#include <iostream>
#include <limits>
#include <math.h>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>

namespace py = pybind11;
using namespace std;

const int32_t LONG_SENTENCE_LEN = 512;


py::array build_location_idx(const py::array_t<int32_t>& sizes_,
			   const py::array_t<int32_t>& doc_idx_,
			   const int32_t seq_length,
			   const int32_t num_epochs,
			   const int64_t tokens_per_epoch) {
    /* Sample index (sample_idx) is used for gpt2 like dataset for which
       the documents are flattened and the samples are built based on this
       1-D flatten array. It is a 2D array with sizes [number-of-samples + 1, 2]
       where [..., 0] contains the index into `doc_idx` and [..., 1] is the
       starting offset in that document.*/

    // Consistency checks.
    assert(seq_length > 1);
    assert(num_epochs > 0);
    assert(tokens_per_epoch > 1);

    // Remove bound checks.
    auto sizes = sizes_.unchecked<1>();
    auto doc_idx = doc_idx_.unchecked<1>();

    // Mapping and it's length (1D).
    int64_t num_samples = (num_epochs * tokens_per_epoch - 1) / seq_length;
    int32_t* sample_idx = new int32_t[2*(num_samples+1)];

    cout << "    using:" << endl << std::flush;
    cout << "     number of sentenses:       " <<
      doc_idx_.shape(0) / num_epochs << endl << std::flush;
    cout << "     number of epochs:          " << num_epochs <<
      endl << std::flush;
    cout << "     sequence length:           " << seq_length <<
      endl << std::flush;
    cout << "     total number of samples:   " << num_samples <<
      endl << std::flush;

    // Index into sample_idx.
    int64_t sample_index = 0;
    // Index into doc_idx.
    int64_t doc_idx_index = 0;
    // Begining offset for each document.
    int32_t doc_offset = 0;
    // Start with first document and no offset.
    sample_idx[2 * sample_index] = doc_idx_index;
    sample_idx[2 * sample_index + 1] = doc_offset;
    ++sample_index;

    while (sample_index <= num_samples) {
        // Start with a fresh sequence.
      int32_t remaining_seq_length = seq_length + 1;
      while (remaining_seq_length != 0) {
            // Get the document length.
	auto doc_id = doc_idx[doc_idx_index];
	auto doc_length = sizes[doc_id] - doc_offset;
	// And add it to the current sequence.
	remaining_seq_length -= doc_length;
	// If we have more than a full sequence, adjust offset and set
	// remaining length to zero so we return from the while loop.
	// Note that -1 here is for the same reason we have -1 in
	// `_num_epochs` calculations.
	if (remaining_seq_length <= 0) {
	  doc_offset += (remaining_seq_length + doc_length - 1);
	  remaining_seq_length = 0;
	} else {
	  // Otherwise, start from the begining of the next document.
	  ++doc_idx_index;
	  doc_offset = 0;
	}
      }
      // Record the sequence.
      sample_idx[2 * sample_index] = doc_idx_index;
      sample_idx[2 * sample_index + 1] = doc_offset;
      ++sample_index;
    }

    // Method to deallocate memory.
    py::capsule free_when_done(sample_idx, [](void *mem_) {
	int32_t *mem = reinterpret_cast<int32_t*>(mem_);
	delete[] mem;
      });

    // Return the numpy array.
    const auto byte_size = sizeof(int32_t);
    return py::array(std::vector<int64_t>{num_samples+1, 2}, // shape
                     {2*byte_size, byte_size}, // C-style contiguous strides
                     sample_idx, // the data pointer
                     free_when_done); // numpy array references
    
}


PYBIND11_MODULE(helpers, m) {
    m.def("build_location_idx", &build_location_idx);
}
