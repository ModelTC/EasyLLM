

def get_comm_times(vit_bs=1,
                   vit_length=1025,
                   vit_hidden_size=3200,
                   llm_bs=1,
                   llm_length=4096,
                   llm_hidden_size=8192,
                   num_layer=149,
                   start_idx=0,
                   vit_layer_num=45,
                   llm_layer_num=48,
                   vit_num_token=256
                   ):

    num_activations = [0] * num_layer
    for i in range(num_layer):
        # to_float, vis embedding increase 2
        if i >= start_idx and i < vit_layer_num + 2:
            num_activations[i] = vit_bs * vit_length * vit_hidden_size
        # mlp projection
        if i == vit_layer_num + 2:
            num_activations[i] = vit_bs * vit_num_token * llm_hidden_size
        # llm activations
        if i >= vit_layer_num + 3:
            num_activations[i] = llm_bs * llm_length * llm_hidden_size
    return num_activations
