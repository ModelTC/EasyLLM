from transformers import LlamaTokenizer
from llm.data.nlp_transforms import SimpleChatParser


def equal(a, b):
    if len(a) != len(b):
        return False
    else:
        for idx in range(len(a)):
            if a[idx] == b[idx]:
                continue
            else:
                return False
    return True


if __name__ == "__main__":
    tokenizer = LlamaTokenizer.from_pretrained("your/tokenizer")
    simple_parser = SimpleChatParser(tokenizer, 4096)

    question = "hello"
    answer = "Hello! How can I help you?"
    system_prompt = "you are a AI assistant. "

    test_meta_1 = {"input": question, "output": answer}
    test_meta_2 = {"system": system_prompt, "input": question, "output": answer}

    test_meta_3 = {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]}
    test_meta_4 = {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}], "system": system_prompt}  # noqa

    test_meta_5 = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    test_meta_6 = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}, {"role": "assistant", "content": answer}]  # noqa

    test_meta_7 = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}, {"role": "user", "content": question}, {"role": "assistant", "content": answer}]  # noqa
    test_meta_8 = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}, {"role": "assistant", "content": answer}, {"role": "user", "content": question}, {"role": "assistant", "content": answer}]  # noqa

    test_input = [test_meta_1, test_meta_3, test_meta_5, test_meta_7]
    test_input_with_system = [test_meta_2, test_meta_4, test_meta_6, test_meta_8]

    test_parser = [simple_parser]
    for parser in test_parser:
        out_1 = parser(test_input[0])['input_ids']
        out_2 = parser(test_input[1])['input_ids']
        out_3 = parser(test_input[2])['input_ids']
        out_4 = parser(test_input[3])['input_ids']
        assert equal(out_1, out_2)
        assert equal(out_1, out_3)
        print(tokenizer.decode(out_1))
        print(tokenizer.decode(out_2))
        print(tokenizer.decode(out_3))
        print(tokenizer.decode(out_4))
        out_sys_1 = parser(test_input_with_system[0])['input_ids']
        out_sys_2 = parser(test_input_with_system[1])['input_ids']
        out_sys_3 = parser(test_input_with_system[2])['input_ids']
        out_sys_4 = parser(test_input_with_system[3])['input_ids']
        assert equal(out_sys_1, out_sys_2)
        assert equal(out_sys_1, out_sys_3)
        print(tokenizer.decode(out_sys_1))
        print(tokenizer.decode(out_sys_2))
        print(tokenizer.decode(out_sys_3))
        print(tokenizer.decode(out_sys_4))
