import os
import json
import torch
import torch.distributed as dist
import pandas as pd
from torch.utils.data import Dataset
try:
    from human_eval.data import read_problems
except: # noqa
    read_problems = None

cmmlu_subject_mapping = {
    'agronomy': '农学',
    'anatomy': '解剖学',
    'ancient_chinese': '古汉语',
    'arts': '艺术学',
    'astronomy': '天文学',
    'business_ethics': '商业伦理',
    'chinese_civil_service_exam': '中国公务员考试',
    'chinese_driving_rule': '中国驾驶规则',
    'chinese_food_culture': '中国饮食文化',
    'chinese_foreign_policy': '中国外交政策',
    'chinese_history': '中国历史',
    'chinese_literature': '中国文学',
    'chinese_teacher_qualification': '中国教师资格',
    'clinical_knowledge': '临床知识',
    'college_actuarial_science': '大学精算学',
    'college_education': '大学教育学',
    'college_engineering_hydrology': '大学工程水文学',
    'college_law': '大学法律',
    'college_mathematics': '大学数学',
    'college_medical_statistics': '大学医学统计',
    'college_medicine': '大学医学',
    'computer_science': '计算机科学',
    'computer_security': '计算机安全',
    'conceptual_physics': '概念物理学',
    'construction_project_management': '建设工程管理',
    'economics': '经济学',
    'education': '教育学',
    'electrical_engineering': '电气工程',
    'elementary_chinese': '小学语文',
    'elementary_commonsense': '小学常识',
    'elementary_information_and_technology': '小学信息技术',
    'elementary_mathematics': '初等数学',
    'ethnology': '民族学',
    'food_science': '食品科学',
    'genetics': '遗传学',
    'global_facts': '全球事实',
    'high_school_biology': '高中生物',
    'high_school_chemistry': '高中化学',
    'high_school_geography': '高中地理',
    'high_school_mathematics': '高中数学',
    'high_school_physics': '高中物理学',
    'high_school_politics': '高中政治',
    'human_sexuality': '人类性行为',
    'international_law': '国际法学',
    'journalism': '新闻学',
    'jurisprudence': '法理学',
    'legal_and_moral_basis': '法律与道德基础',
    'logical': '逻辑学',
    'machine_learning': '机器学习',
    'management': '管理学',
    'marketing': '市场营销',
    'marxist_theory': '马克思主义理论',
    'modern_chinese': '现代汉语',
    'nutrition': '营养学',
    'philosophy': '哲学',
    'professional_accounting': '专业会计',
    'professional_law': '专业法学',
    'professional_medicine': '专业医学',
    'professional_psychology': '专业心理学',
    'public_relations': '公共关系',
    'security_study': '安全研究',
    'sociology': '社会学',
    'sports_science': '体育学',
    'traditional_chinese_medicine': '中医中药',
    'virology': '病毒学',
    'world_history': '世界历史',
    'world_religions': '世界宗教'
}


ceval_subject_mapping = {
    "computer_network":
    ["Computer Network", "\u8ba1\u7b97\u673a\u7f51\u7edc", "STEM"],
    "operating_system":
    ["Operating System", "\u64cd\u4f5c\u7cfb\u7edf", "STEM"],
    "computer_architecture":
    ["Computer Architecture", "\u8ba1\u7b97\u673a\u7ec4\u6210", "STEM"],
    "college_programming":
    ["College Programming", "\u5927\u5b66\u7f16\u7a0b", "STEM"],
    "college_physics": ["College Physics", "\u5927\u5b66\u7269\u7406", "STEM"],
    "college_chemistry":
    ["College Chemistry", "\u5927\u5b66\u5316\u5b66", "STEM"],
    "advanced_mathematics":
    ["Advanced Mathematics", "\u9ad8\u7b49\u6570\u5b66", "STEM"],
    "probability_and_statistics":
    ["Probability and Statistics", "\u6982\u7387\u7edf\u8ba1", "STEM"],
    "discrete_mathematics":
    ["Discrete Mathematics", "\u79bb\u6563\u6570\u5b66", "STEM"],
    "electrical_engineer": [
        "Electrical Engineer", "\u6ce8\u518c\u7535\u6c14\u5de5\u7a0b\u5e08",
        "STEM"
    ],
    "metrology_engineer":
    ["Metrology Engineer", "\u6ce8\u518c\u8ba1\u91cf\u5e08", "STEM"],
    "high_school_mathematics":
    ["High School Mathematics", "\u9ad8\u4e2d\u6570\u5b66", "STEM"],
    "high_school_physics":
    ["High School Physics", "\u9ad8\u4e2d\u7269\u7406", "STEM"],
    "high_school_chemistry":
    ["High School Chemistry", "\u9ad8\u4e2d\u5316\u5b66", "STEM"],
    "high_school_biology": [
        "High School Biology", "\u9ad8\u4e2d\u751f\u7269", "STEM"
    ],
    "middle_school_mathematics": [
        "Middle School Mathematics", "\u521d\u4e2d\u6570\u5b66", "STEM"
    ],
    "middle_school_biology": [
        "Middle School Biology", "\u521d\u4e2d\u751f\u7269", "STEM"
    ],
    "middle_school_physics": [
        "Middle School Physics", "\u521d\u4e2d\u7269\u7406", "STEM"
    ],
    "middle_school_chemistry": [
        "Middle School Chemistry", "\u521d\u4e2d\u5316\u5b66", "STEM"
    ],
    "veterinary_medicine": [
        "Veterinary Medicine", "\u517d\u533b\u5b66", "STEM"
    ],
    "college_economics": [
        "College Economics", "\u5927\u5b66\u7ecf\u6d4e\u5b66", "Social Science"
    ],
    "business_administration": [
        "Business Administration", "\u5de5\u5546\u7ba1\u7406", "Social Science"
    ],
    "marxism": [
        "Marxism", "\u9a6c\u514b\u601d\u4e3b\u4e49\u57fa\u672c\u539f\u7406",
        "Social Science"
    ],
    "mao_zedong_thought": [
        "Mao Zedong Thought",
        "\u6bdb\u6cfd\u4e1c\u601d\u60f3\u548c\u4e2d\u56fd\u7279\u8272\u793e\u4f1a\u4e3b\u4e49\u7406\u8bba\u4f53\u7cfb\u6982\u8bba", # noqa
        "Social Science"
    ],
    "education_science": [
        "Education Science", "\u6559\u80b2\u5b66", "Social Science"
    ],
    "teacher_qualification": [
        "Teacher Qualification", "\u6559\u5e08\u8d44\u683c", "Social Science"
    ],
    "high_school_politics": [
        "High School Politics", "\u9ad8\u4e2d\u653f\u6cbb", "Social Science"
    ],
    "high_school_geography": [
        "High School Geography", "\u9ad8\u4e2d\u5730\u7406", "Social Science"
    ],
    "middle_school_politics": [
        "Middle School Politics", "\u521d\u4e2d\u653f\u6cbb", "Social Science"
    ],
    "middle_school_geography": [
        "Middle School Geography", "\u521d\u4e2d\u5730\u7406", "Social Science"
    ],
    "modern_chinese_history":
    ["Modern Chinese History", "\u8fd1\u4ee3\u53f2\u7eb2\u8981", "Humanities"],
    "ideological_and_moral_cultivation": [
        "Ideological and Moral Cultivation",
        "\u601d\u60f3\u9053\u5fb7\u4fee\u517b\u4e0e\u6cd5\u5f8b\u57fa\u7840",
        "Humanities"
    ],
    "logic": ["Logic", "\u903b\u8f91\u5b66", "Humanities"],
    "law": ["Law", "\u6cd5\u5b66", "Humanities"],
    "chinese_language_and_literature": [
        "Chinese Language and Literature",
        "\u4e2d\u56fd\u8bed\u8a00\u6587\u5b66", "Humanities"
    ],
    "art_studies": ["Art Studies", "\u827a\u672f\u5b66", "Humanities"],
    "professional_tour_guide": [
        "Professional Tour Guide", "\u5bfc\u6e38\u8d44\u683c", "Humanities"
    ],
    "legal_professional": [
        "Legal Professional", "\u6cd5\u5f8b\u804c\u4e1a\u8d44\u683c",
        "Humanities"
    ],
    "high_school_chinese": [
        "High School Chinese", "\u9ad8\u4e2d\u8bed\u6587", "Humanities"
    ],
    "high_school_history": [
        "High School History", "\u9ad8\u4e2d\u5386\u53f2", "Humanities"
    ],
    "middle_school_history": [
        "Middle School History", "\u521d\u4e2d\u5386\u53f2", "Humanities"
    ],
    "civil_servant": ["Civil Servant", "\u516c\u52a1\u5458", "Other"],
    "sports_science": ["Sports Science", "\u4f53\u80b2\u5b66", "Other"],
    "plant_protection": [
        "Plant Protection", "\u690d\u7269\u4fdd\u62a4", "Other"
    ],
    "basic_medicine": ["Basic Medicine", "\u57fa\u7840\u533b\u5b66", "Other"],
    "clinical_medicine": [
        "Clinical Medicine", "\u4e34\u5e8a\u533b\u5b66", "Other"
    ],
    "urban_and_rural_planner": [
        "Urban and Rural Planner",
        "\u6ce8\u518c\u57ce\u4e61\u89c4\u5212\u5e08", "Other"
    ],
    "accountant": ["Accountant", "\u6ce8\u518c\u4f1a\u8ba1\u5e08", "Other"],
    "fire_engineer": [
        "Fire Engineer", "\u6ce8\u518c\u6d88\u9632\u5de5\u7a0b\u5e08", "Other"
    ],
    "environmental_impact_assessment_engineer": [
        "Environmental Impact Assessment Engineer",
        "\u73af\u5883\u5f71\u54cd\u8bc4\u4ef7\u5de5\u7a0b\u5e08", "Other"
    ],
    "tax_accountant": ["Tax Accountant", "\u7a0e\u52a1\u5e08", "Other"],
    "physician": ["Physician", "\u533b\u5e08\u8d44\u683c", "Other"]
}


class EvalDataset(Dataset):
    def __init__(self, task_type, question_dir=None, load_type="line"):
        super(EvalDataset, self).__init__()
        self.task_type = task_type
        self.question_dir = question_dir
        self.load_type = load_type
        self.load_dataset()

    def load_dataset(self):
        self.problems, self.dev_problems = {}, {}
        self.idx2subproj = {}
        if self.task_type == "human_eval":
            # 'HumanEval/0'
            #    - ['task_id', 'prompt', 'entry_point', 'canonical_solution', 'test']
            self.problems = read_problems()
            keys = list(self.problems.keys())
            for idx in range(len(keys)):
                self.idx2subproj[idx] = keys[idx]
        elif self.task_type == "base":
            assert os.path.isfile(self.question_dir), f"{self.question_dir} is not a json file"
            with open(self.question_dir, "r") as f:
                if self.load_type == "line":
                    metas = f.readlines()
                    metas = [json.loads(meta.strip()) for meta in metas]
                elif self.load_type == "all":
                    metas = json.load(f)
                for task_id in range(len(metas)):
                    item = {}
                    data = metas[task_id]
                    instruction = data.get("instruction", "")
                    input_text = data.get("input", "")
                    item["question"] = instruction + input_text
                    item["answer"] = data["output"]
                    self.problems[task_id] = item
        elif self.task_type in ["cmmlu", "ceval"]:
            assert self.question_dir is not None, "question dir is none!"
            assert os.path.exists(self.question_dir), "question dir does not exist."
            question_phase, answer_phase = "Question", "Answer"
            if self.task_type in ["cmmlu"]:
                test_dir, dev_dir = os.path.join(self.question_dir, "test"), os.path.join(self.question_dir, "dev")
                question_phase, answer_phase = "Question", "Answer"
            elif self.task_type in ["ceval"]:
                test_dir, dev_dir = os.path.join(self.question_dir, "val"), os.path.join(self.question_dir, "dev")
                question_phase, answer_phase = "question", "answer"
            assert os.path.exists(test_dir), f"{test_dir} does not exist."
            assert os.path.exists(dev_dir), f"{dev_dir} does not exist."
            # load test dataset
            entire_idx = 0
            for _, _, files in os.walk(test_dir):
                for file_name in files:
                    file_path = os.path.join(test_dir, file_name)
                    df = pd.read_csv(file_path)
                    for idx, row in df.iterrows():
                        if self.task_type in ["cmmlu"]:
                            task_id = f"{file_name.split('.')[0]}/{idx}"
                        elif self.task_type in ["ceval"]:
                            task_phase = file_name.split('.')[0]
                            task_phase = "_".join(task_phase.split("_")[:-1])
                            task_id = f"{task_phase}/{idx}"
                        self.problems[task_id] = {}
                        self.problems[task_id]["task_id"] = task_id
                        self.problems[task_id]["question"] = row[question_phase]
                        self.problems[task_id]["A"] = row["A"]
                        self.problems[task_id]["B"] = row["B"]
                        self.problems[task_id]["C"] = row["C"]
                        self.problems[task_id]["D"] = row["D"]
                        self.problems[task_id]["answer"] = row[answer_phase]
                        self.idx2subproj[entire_idx] = task_id
                        entire_idx += 1
            # process dev dataset prompt
            for _, _, files in os.walk(dev_dir):
                for file_name in files:
                    if self.task_type in ["cmmlu"]:
                        subject_phase = file_name.split(".")[0]
                    elif self.task_type in ["ceval"]:
                        subject_phase = file_name.split('.')[0]
                        subject_phase = "_".join(subject_phase.split("_")[:-1])
                    if self.task_type in ["cmmlu"]:
                        _ch_name = cmmlu_subject_mapping[subject_phase]
                    elif self.task_type in ["ceval"]:
                        _ch_name = ceval_subject_mapping[subject_phase][1]
                    file_path = os.path.join(dev_dir, file_name)
                    df = pd.read_csv(file_path, nrows=5)
                    prompt = ""
                    for idx, row in df.iterrows():
                        question = row[question_phase]
                        A, B, C, D = row["A"], row["B"], row["C"], row["D"]
                        answer = row[answer_phase]
                        if self.task_type in ["cmmlu"]:
                            prompt += f"以下是关于{_ch_name}的单项选择题，请直接给出正确答案的选项。\n题目：{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案是: {answer}\n" # noqa
                        elif self.task_type in ["ceval"]:
                            prompt += f"以下是中国关于{_ch_name}考试的单项选择题，请选出其中的正确答案。\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案: {answer}\n" # noqa
                    self.dev_problems[subject_phase] = {}
                    self.dev_problems[subject_phase]["prompt"] = prompt
                    self.dev_problems[subject_phase]["_ch_name"] = _ch_name

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        task_id, subject_phase = idx, "none"
        if self.task_type in ["cmmlu", "ceval", "human_eval"]:
            task_id = self.idx2subproj[idx]
            subject_phase = self.problems[task_id]["task_id"].split("/")[0]
        return task_id, self.problems[task_id], self.dev_problems.get(subject_phase, {})


class LocalEvalDataset(object):
    def __init__(self, base_dataset):
        super(LocalEvalDataset, self).__init__()
        self.base_dataset = base_dataset

    def generate_prompt(self, problem, dev_problem):
        if self.base_dataset.task_type == "human_eval":
            input_text = problem["prompt"]
            answer = ""
        elif self.base_dataset.task_type == "base":
            input_text = problem["question"]
            answer = problem["answer"]
        elif self.base_dataset.task_type in ["cmmlu", "ceval"]:
            # task_id = problem["task_id"]
            # subject_phase = task_id.split("/")[0]
            pre_prompt = dev_problem["prompt"]
            _ch_name = dev_problem["_ch_name"]

            question = problem["question"]
            A, B, C, D = problem["A"], problem["B"], problem["C"], problem["D"]
            answer = problem["answer"]
            if self.base_dataset.task_type in ["cmmlu"]:
                input_text = pre_prompt + \
                    f"以下是关于{_ch_name}的单项选择题，请直接给出正确答案的选项。\n题目：{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案是: "
            elif self.base_dataset.task_type in ["ceval"]:
                input_text = pre_prompt + \
                    f"以下是中国关于{_ch_name}考试的单项选择题，请选出其中的正确答案。\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案: "
        return input_text, answer

    def get_data(self, idx):
        task_id, problem, dev_problem = self.base_dataset[idx]
        input_text, answer = self.generate_prompt(problem, dev_problem)
        return task_id, input_text, answer


class SampleEvalDataset(object):
    def __init__(self, base_dataset, num_replicas=None, rank=None):
        super(SampleEvalDataset, self).__init__()
        self.base_dataset = base_dataset
        if num_replicas is None:
            num_replicas = dist.get_world_size()  # os.environ["WORLD_SIZE"]
        if rank is None:
            rank = dist.get_rank()  # os.environ["LOCAL_RANK"]

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = len(range(rank, len(self.base_dataset), num_replicas))
        self.total_size = len(self.base_dataset)
        self.sample_ids = self.get_sample_ids()

    def get_sample_ids(self):
        indices = torch.arange(len(self.base_dataset))
        indices = indices[self.rank::self.num_replicas].numpy().tolist()
        # indices = set(indices)
        assert len(indices) == self.num_samples
        return indices

    def __len__(self):
        return self.num_samples

    def generate_prompt(self, problem, dev_problem):
        if self.base_dataset.task_type == "human_eval":
            input_text = problem["prompt"]
            answer = ""
        elif self.base_dataset.task_type == "base":
            input_text = problem["question"]
            answer = problem["answer"]
        elif self.base_dataset.task_type in ["cmmlu", "ceval"]:
            # task_id = problem["task_id"]
            # subject_phase = task_id.split("/")[0]
            pre_prompt = dev_problem["prompt"]
            _ch_name = dev_problem["_ch_name"]

            question = problem["question"]
            A, B, C, D = problem["A"], problem["B"], problem["C"], problem["D"]
            answer = problem["answer"]
            if self.base_dataset.task_type in ["cmmlu"]:
                input_text = pre_prompt + \
                    f"以下是关于{_ch_name}的单项选择题，请直接给出正确答案的选项。\n题目：{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案是: "
            elif self.base_dataset.task_type in ["ceval"]:
                input_text = pre_prompt + \
                    f"以下是中国关于{_ch_name}考试的单项选择题，请选出其中的正确答案。\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案: "
        return input_text, answer

    def get_items(self):
        input_texts = []
        for idx in range(len(self.sample_ids)):
            sample_id = self.sample_ids[idx]
            task_id, problem, dev_problem = self.base_dataset[sample_id]
            input_text, answer = self.generate_prompt(problem, dev_problem)
            if self.base_dataset.task_type == "human_eval":
                input_texts.append((task_id, input_text, answer))
            else:
                input_texts.append((sample_id, input_text, answer))
        return iter(input_texts)
